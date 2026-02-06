import { SecretsManagerClient, GetSecretValueCommand } from '@aws-sdk/client-secrets-manager';
import crypto from 'crypto';

const secretsClient = new SecretsManagerClient();
let cachedSecrets = null;

/**
 * Load secrets from Secrets Manager (cached for container lifetime)
 */
async function getSecrets() {
  if (cachedSecrets) {
    return cachedSecrets;
  }

  const command = new GetSecretValueCommand({
    SecretId: process.env.SECRET_NAME,
  });

  const response = await secretsClient.send(command);
  if (!response.SecretString) {
    throw new Error('Empty secret returned from Secrets Manager');
  }
  cachedSecrets = JSON.parse(response.SecretString);

  // Validate required secrets
  const requiredSecrets = ['slack_signing_secret', 'slack_bot_token', 'anthropic_api_key', 'mac_tailscale_funnel_hostname', 'openclaw_gateway_token'];
  for (const secret of requiredSecrets) {
    if (!cachedSecrets[secret]) {
      throw new Error(`Missing required secret: ${secret}`);
    }
  }

  return cachedSecrets;
}

/**
 * Timing-safe HMAC comparison
 */
function timingSafeEqual(a, b) {
  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);

  if (bufA.length !== bufB.length) {
    return false;
  }

  let result = 0;
  for (let i = 0; i < bufA.length; i++) {
    result |= bufA[i] ^ bufB[i];
  }
  return result === 0;
}

/**
 * Verify Slack request signature
 * https://api.slack.com/authentication/verifying-requests-from-slack
 */
function verifySlackSignature(body, timestamp, slackSignature, signingSecret) {
  // Check timestamp is within 5 minutes
  const currentTime = Math.floor(Date.now() / 1000);
  if (Math.abs(currentTime - timestamp) > 300) {
    return false;
  }

  // Create signature
  const baseString = `v0:${timestamp}:${body}`;
  const computed = `v0=${crypto
    .createHmac('sha256', signingSecret)
    .update(baseString)
    .digest('hex')}`;

  return timingSafeEqual(computed, slackSignature);
}

/**
 * Check if Mac gateway is reachable
 */
async function isMacGatewayOnline(hostname, token) {
  try {
    const url = `https://${hostname}:18789/`;
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
      timeout: 5000,
    });
    return response.ok;
  } catch (error) {
    console.log(`Mac gateway offline: ${error.message}`);
    return false;
  }
}

/**
 * Call Mac gateway for completion
 */
async function callMacGateway(hostname, token, prompt) {
  const url = `https://${hostname}:18789/v1/chat/completions`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'claude',
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 1000,
    }),
    timeout: 30000,
  });

  if (!response.ok) {
    throw new Error(`Mac gateway error: ${response.status}`);
  }

  let data;
  try {
    data = await response.json();
  } catch (parseError) {
    console.error('Failed to parse Mac gateway response:', parseError);
    throw new Error(`Failed to parse Mac gateway response: ${parseError.message}`);
  }

  console.log('Mac gateway response:', JSON.stringify(data));

  if (!data || typeof data !== 'object') {
    console.error('Invalid Mac gateway response: not an object:', JSON.stringify(data));
    throw new Error(`Invalid response from Mac gateway: expected object`);
  }

  if (!Array.isArray(data.choices) || !data.choices[0]) {
    console.error('Invalid Mac gateway response:', JSON.stringify(data));
    throw new Error(`Invalid response from Mac gateway: missing or invalid choices array`);
  }

  const choice = data.choices[0];
  if (!choice || typeof choice !== 'object' || !choice.message || typeof choice.message !== 'object' || !choice.message.content) {
    console.error('Invalid Mac gateway response - no message:', JSON.stringify(data));
    throw new Error(`Invalid response from Mac gateway: missing message content`);
  }

  return choice.message.content;
}

/**
 * Call Anthropic API directly
 */
async function callAnthropicAPI(apiKey, prompt) {
  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'content-type': 'application/json',
    },
    body: JSON.stringify({
      model: 'claude-opus-4-6',
      max_tokens: 1000,
      messages: [{ role: 'user', content: prompt }],
    }),
    timeout: 30000,
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Anthropic API error: ${response.status} ${error}`);
  }

  let data;
  try {
    data = await response.json();
  } catch (parseError) {
    console.error('Failed to parse Anthropic API response:', parseError);
    throw new Error(`Failed to parse Anthropic API response: ${parseError.message}`);
  }

  console.log('Anthropic API response:', JSON.stringify(data));

  if (!data || typeof data !== 'object') {
    console.error('Invalid Anthropic API response: not an object:', JSON.stringify(data));
    throw new Error(`Invalid response from Anthropic API: expected object`);
  }

  if (!Array.isArray(data.content) || !data.content[0]) {
    console.error('Invalid Anthropic API response:', JSON.stringify(data));
    throw new Error(`Invalid response from Anthropic API: missing or invalid content array`);
  }

  const content = data.content[0];
  if (!content || typeof content !== 'object' || !content.text) {
    console.error('Invalid Anthropic API response - no text:', JSON.stringify(data));
    throw new Error(`Invalid response from Anthropic API: missing text in content`);
  }

  return content.text;
}

/**
 * Post message to Slack channel
 */
async function postSlackMessage(botToken, channelId, text) {
  const response = await fetch('https://slack.com/api/chat.postMessage', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${botToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      channel: channelId,
      text: text,
    }),
  });

  if (!response.ok) {
    throw new Error(`Slack API HTTP error: ${response.status}`);
  }

  let data;
  try {
    data = await response.json();
  } catch (parseError) {
    console.error('Failed to parse Slack API response:', parseError);
    throw new Error(`Failed to parse Slack API response: ${parseError.message}`);
  }

  if (!data || typeof data !== 'object') {
    console.error('Invalid Slack API response: not an object:', JSON.stringify(data));
    throw new Error(`Slack API error: invalid response format`);
  }

  if (!data.ok) {
    const errorMsg = data.error || 'Unknown error';
    throw new Error(`Slack API error: ${errorMsg}`);
  }
}

/**
 * Send Slack response
 */
async function sendSlackResponse(botToken, responseUrl, text) {
  try {
    if (!responseUrl) {
      console.error('No responseUrl provided');
      return;
    }

    const response = await fetch(responseUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        response_type: 'in_channel',
        text: text,
      }),
    });

    if (!response.ok) {
      console.error(`Failed to send response: ${response.status} ${response.statusText}`);
    }
  } catch (error) {
    console.error('Error sending Slack response:', error.message);
  }
}

/**
 * Handle health check
 */
function handleHealth(event) {
  return {
    statusCode: 200,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      status: 'ok',
      timestamp: new Date().toISOString(),
    }),
  };
}

/**
 * Handle slash command
 */
async function handleSlashCommand(event, secrets) {
  let params;
  try {
    const body = event.body;
    console.log('Slash command body:', body);
    if (!body) {
      throw new Error('Empty request body');
    }
    params = new URLSearchParams(body);
    console.log('Parsed params keys:', Array.from(params.keys()));
  } catch (error) {
    console.error('Failed to parse slash command body:', error);
    return {
      statusCode: 400,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: 'Invalid request body' }),
    };
  }

  const responseUrl = params.get('response_url');
  const userId = params.get('user_id');
  const text = params.get('text') || 'hello';
  const channelId = params.get('channel_id');

  console.log('Extracted slash command params:', {
    responseUrl: responseUrl ? '***' : 'MISSING',
    userId,
    text,
    channelId,
  });

  if (!responseUrl) {
    console.error('Missing response_url in slash command');
    return {
      statusCode: 400,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: 'Missing response_url' }),
    };
  }

  // Acknowledge immediately
  const ackResponse = {
    statusCode: 200,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: 'Thinking...' }),
  };

  // Process async (in practice, this needs to be done via SQS or similar in production)
  (async () => {
    try {
      console.log('Processing slash command - text:', text, 'userId:', userId, 'channelId:', channelId);

      const macOnline = await isMacGatewayOnline(
        secrets.mac_tailscale_funnel_hostname,
        secrets.openclaw_gateway_token
      );

      console.log('Mac gateway online:', macOnline);

      let result;
      if (macOnline) {
        result = await callMacGateway(
          secrets.mac_tailscale_funnel_hostname,
          secrets.openclaw_gateway_token,
          text
        );
      } else {
        result = `(Mac offline) ${await callAnthropicAPI(secrets.anthropic_api_key, text)}`;
      }

      console.log('Got result:', result);

      await sendSlackResponse(
        secrets.slack_bot_token,
        responseUrl,
        result
      );
    } catch (error) {
      console.error('Slash command error:', error);
      console.error('Error stack:', error.stack);
      await sendSlackResponse(
        secrets.slack_bot_token,
        responseUrl,
        `Error: ${error.message}`
      );
    }
  })().catch(error => {
    console.error('Async error:', error);
    console.error('Async error stack:', error.stack);
  });

  return ackResponse;
}

/**
 * Handle Events API
 */
async function handleEvents(event, secrets) {
  console.log('=== handleEvents called ===');
  console.log('Event body type:', typeof event.body);
  console.log('Event body length:', event.body ? event.body.length : 'NONE');

  let body;
  try {
    body = JSON.parse(event.body);
    console.log('Parsed body type:', body.type);
  } catch (error) {
    console.error('Failed to parse event body:', error);
    console.error('Raw body sample:', event.body ? event.body.substring(0, 200) : 'NONE');
    return {
      statusCode: 400,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: 'Invalid JSON in request body' }),
    };
  }

  // Handle url_verification challenge
  if (body.type === 'url_verification') {
    return {
      statusCode: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ challenge: body.challenge }),
    };
  }

  // Ack immediately
  const ackResponse = {
    statusCode: 200,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  };

  // Process async
  if (body.event) {
    (async () => {
      try {
        const eventType = body.event.type;
        const text = body.event.text;
        const channel = body.event.channel;
        const user = body.event.user;

        if (!text || !channel) {
          return;
        }

        const macOnline = await isMacGatewayOnline(
          secrets.mac_tailscale_funnel_hostname,
          secrets.openclaw_gateway_token
        );

        let result;
        if (macOnline) {
          result = await callMacGateway(
            secrets.mac_tailscale_funnel_hostname,
            secrets.openclaw_gateway_token,
            text
          );
        } else {
          result = `(Mac offline) ${await callAnthropicAPI(secrets.anthropic_api_key, text)}`;
        }

        await postSlackMessage(
          secrets.slack_bot_token,
          channel,
          result
        );
      } catch (error) {
        console.error('Events handling error:', error);
        console.error('Error message:', error.message);
        console.error('Error stack:', error.stack);
      }
    })().catch(error => {
      console.error('Async error:', error);
      console.error('Async error stack:', error.stack);
    });
  }

  return ackResponse;
}

/**
 * Get method from event (handles both API Gateway v1 and v2)
 */
function getMethod(event) {
  // API Gateway v2 format
  if (event.requestContext?.http?.method) {
    return event.requestContext.http.method;
  }
  // API Gateway v1 format
  if (event.requestContext?.httpMethod) {
    return event.requestContext.httpMethod;
  }
  // Fallback
  return event.httpMethod || event.method || 'GET';
}

/**
 * Get path from event (handles both API Gateway v1 and v2)
 */
function getPath(event) {
  // API Gateway v2 format
  if (event.requestContext?.http?.path) {
    return event.requestContext.http.path;
  }
  // API Gateway v1 format
  if (event.requestContext?.path) {
    return event.requestContext.path;
  }
  // Fallback
  return event.path || '/';
}

/**
 * Get headers from event (handles both API Gateway v1 and v2, and Lambda Function URLs)
 */
function getHeaders(event) {
  // Handle case-insensitive header access
  const headers = event.headers || event.multiValueHeaders || {};
  return headers;
}

/**
 * Main Lambda handler
 */
export async function handler(event) {
  console.log('Received event:', JSON.stringify(event));

  try {
    // Get raw body for signature verification
    let rawBody = event.rawBody || event.body;

    // Decode if base64 encoded
    if (event.isBase64Encoded && rawBody) {
      rawBody = Buffer.from(rawBody, 'base64').toString('utf-8');
    }

    console.log('Raw body:', rawBody);

    // Get secrets
    const secrets = await getSecrets();

    const method = getMethod(event);
    const path = getPath(event);
    const headers = getHeaders(event);

    // Verify Slack signature for POST requests
    if (method === 'POST') {
      // Create case-insensitive header lookup
      const headerLookup = {};
      for (const [key, value] of Object.entries(headers || {})) {
        headerLookup[key.toLowerCase()] = value;
      }

      const slackSignature = headerLookup['x-slack-signature'];
      const slackTimestamp = headerLookup['x-slack-request-timestamp'];

      if (!slackSignature || !slackTimestamp) {
        console.error('Missing Slack signature headers. Headers:', Object.keys(headers || {}));
        return {
          statusCode: 401,
          body: JSON.stringify({ error: 'Missing Slack signature headers' }),
        };
      }

      if (!verifySlackSignature(rawBody, parseInt(slackTimestamp), slackSignature, secrets.slack_signing_secret)) {
        console.error('Invalid Slack signature');
        return {
          statusCode: 401,
          body: JSON.stringify({ error: 'Invalid signature' }),
        };
      }
    }

    // Route to handler
    if (path === '/health' && method === 'GET') {
      return handleHealth(event);
    } else if (path === '/slack/slash' && method === 'POST') {
      return await handleSlashCommand(event, secrets);
    } else if (path === '/slack/events' && method === 'POST') {
      return await handleEvents(event, secrets);
    } else {
      return {
        statusCode: 404,
        body: JSON.stringify({ error: 'Not found' }),
      };
    }
  } catch (error) {
    console.error('Handler error:', error);
    return {
      statusCode: 500,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: error.message }),
    };
  }
}
