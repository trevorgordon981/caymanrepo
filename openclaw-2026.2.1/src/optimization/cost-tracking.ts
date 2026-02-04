/**
 * Cost Tracking & Analytics Module
 * Logs every optimization and cost savings
 * Provides real-time dashboard metrics
 */

export interface OptimizationEvent {
  id: string;
  timestamp: number;
  feature: string; // truncation, compression, routing, batching, context
  tokensOriginal: number;
  tokensOptimized: number;
  tokensSaved: number;
  costOriginal: number;
  costOptimized: number;
  costSaved: number;
}

export interface CostMetrics {
  totalEvents: number;
  totalTokensSaved: number;
  totalCostSaved: number;
  byFeature: Record<string, { events: number; tokensSaved: number; costSaved: number }>;
  hourlyBreakdown: Record<string, { count: number; savings: number }>;
  daily?: { date: string; cost: number; tokens: number }[];
  monthly?: { month: string; cost: number; tokens: number }[];
  yearly?: { year: string; cost: number; tokens: number };
  monthlyCurrent?: number;
  monthlyProjected?: number;
  yearlyProjected?: number;
}

export class CostTracker {
  private events: OptimizationEvent[] = [];
  private sessionStartTime: number = Date.now();

  constructor(private haikuCost: number = 0.0008, private opusCost: number = 0.015) {}

  /**
   * Record an optimization event
   */
  recordEvent(
    feature: string,
    tokensOriginal: number,
    tokensOptimized: number
  ): OptimizationEvent {
    const tokensSaved = tokensOriginal - tokensOptimized;

    // Estimate costs (roughly $0.0008 per 1000 tokens for Haiku)
    const costOriginal = (tokensOriginal / 1000) * this.haikuCost;
    const costOptimized = (tokensOptimized / 1000) * this.haikuCost;
    const costSaved = costOriginal - costOptimized;

    const event: OptimizationEvent = {
      id: `evt_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
      timestamp: Date.now(),
      feature,
      tokensOriginal,
      tokensOptimized,
      tokensSaved,
      costOriginal,
      costOptimized,
      costSaved,
    };

    this.events.push(event);
    return event;
  }

  /**
   * Get metrics including daily, monthly, and yearly aggregations
   */
  getMetrics(): CostMetrics {
    const byFeature: Record<string, { events: number; tokensSaved: number; costSaved: number }> = {};
    const hourlyBreakdown: Record<string, { count: number; savings: number }> = {};
    const dailyBreakdown: Record<string, { cost: number; tokens: number }> = {};
    const monthlyBreakdown: Record<string, { cost: number; tokens: number }> = {};
    const yearlyBreakdown: Record<string, { cost: number; tokens: number }> = {};

    let totalTokensSaved = 0;
    let totalCostSaved = 0;

    for (const event of this.events) {
      totalTokensSaved += event.tokensSaved;
      totalCostSaved += event.costSaved;

      // By feature
      if (!byFeature[event.feature]) {
        byFeature[event.feature] = { events: 0, tokensSaved: 0, costSaved: 0 };
      }
      byFeature[event.feature].events++;
      byFeature[event.feature].tokensSaved += event.tokensSaved;
      byFeature[event.feature].costSaved += event.costSaved;

      // Hourly breakdown
      const hour = new Date(event.timestamp).toISOString().slice(0, 13);
      if (!hourlyBreakdown[hour]) {
        hourlyBreakdown[hour] = { count: 0, savings: 0 };
      }
      hourlyBreakdown[hour].count++;
      hourlyBreakdown[hour].savings += event.costSaved;

      // Daily breakdown
      const date = new Date(event.timestamp).toISOString().slice(0, 10);
      if (!dailyBreakdown[date]) {
        dailyBreakdown[date] = { cost: 0, tokens: 0 };
      }
      dailyBreakdown[date].cost += event.costSaved;
      dailyBreakdown[date].tokens += event.tokensSaved;

      // Monthly breakdown
      const month = new Date(event.timestamp).toISOString().slice(0, 7);
      if (!monthlyBreakdown[month]) {
        monthlyBreakdown[month] = { cost: 0, tokens: 0 };
      }
      monthlyBreakdown[month].cost += event.costSaved;
      monthlyBreakdown[month].tokens += event.tokensSaved;

      // Yearly breakdown
      const year = new Date(event.timestamp).toISOString().slice(0, 4);
      if (!yearlyBreakdown[year]) {
        yearlyBreakdown[year] = { cost: 0, tokens: 0 };
      }
      yearlyBreakdown[year].cost += event.costSaved;
      yearlyBreakdown[year].tokens += event.tokensSaved;
    }

    // Calculate current month and projections
    const now = new Date();
    const currentMonth = now.toISOString().slice(0, 7);
    const currentYear = now.toISOString().slice(0, 4);
    const daysInMonth = new Date(now.getFullYear(), now.getMonth() + 1, 0).getDate();
    const currentDay = now.getDate();

    const monthlyCurrent = monthlyBreakdown[currentMonth]?.cost || 0;
    const monthlyProjected = (monthlyCurrent / currentDay) * daysInMonth;
    const yearlyProjected = monthlyProjected * 12;

    // Convert maps to arrays
    const daily = Object.entries(dailyBreakdown).map(([date, data]) => ({
      date,
      cost: data.cost,
      tokens: data.tokens,
    }));

    const monthly = Object.entries(monthlyBreakdown).map(([month, data]) => ({
      month,
      cost: data.cost,
      tokens: data.tokens,
    }));

    const yearly = yearlyBreakdown[currentYear] || { cost: 0, tokens: 0 };

    return {
      totalEvents: this.events.length,
      totalTokensSaved,
      totalCostSaved,
      byFeature,
      hourlyBreakdown,
      daily,
      monthly,
      yearly: { year: currentYear, cost: yearly.cost, tokens: yearly.tokens },
      monthlyCurrent,
      monthlyProjected,
      yearlyProjected,
    };
  }

  /**
   * Get session summary
   */
  getSessionSummary(): {
    duration: number;
    events: number;
    tokensSaved: number;
    costSaved: number;
    averagePerEvent: { tokens: number; cost: number };
  } {
    const metrics = this.getMetrics();
    const duration = Date.now() - this.sessionStartTime;
    const avgTokens = metrics.totalTokensSaved / Math.max(metrics.totalEvents, 1);
    const avgCost = metrics.totalCostSaved / Math.max(metrics.totalEvents, 1);

    return {
      duration,
      events: metrics.totalEvents,
      tokensSaved: metrics.totalTokensSaved,
      costSaved: metrics.totalCostSaved,
      averagePerEvent: { tokens: avgTokens, cost: avgCost },
    };
  }

  /**
   * Export events as CSV
   */
  exportCSV(): string {
    if (this.events.length === 0) {
      return "id,timestamp,feature,tokensOriginal,tokensOptimized,tokensSaved,costOriginal,costOptimized,costSaved";
    }

    const header =
      "id,timestamp,feature,tokensOriginal,tokensOptimized,tokensSaved,costOriginal,costOptimized,costSaved";
    const rows = this.events.map((evt) => {
      return [
        evt.id,
        new Date(evt.timestamp).toISOString(),
        evt.feature,
        evt.tokensOriginal,
        evt.tokensOptimized,
        evt.tokensSaved,
        evt.costOriginal.toFixed(6),
        evt.costOptimized.toFixed(6),
        evt.costSaved.toFixed(6),
      ].join(",");
    });

    return [header, ...rows].join("\n");
  }

  /**
   * Export as JSON
   */
  exportJSON(): {
    metadata: { exportedAt: string; sessionStartTime: string };
    metrics: CostMetrics;
    events: OptimizationEvent[];
  } {
    return {
      metadata: {
        exportedAt: new Date().toISOString(),
        sessionStartTime: new Date(this.sessionStartTime).toISOString(),
      },
      metrics: this.getMetrics(),
      events: this.events,
    };
  }

  /**
   * Clear history (useful for testing)
   */
  clear(): void {
    this.events = [];
    this.sessionStartTime = Date.now();
  }
}

/**
 * Format cost for display
 */
export function formatCost(cost: number): string {
  if (cost < 0.001) {
    return `$${(cost * 1000000).toFixed(2)}µ`; // microdollars
  }
  if (cost < 0.01) {
    return `$${(cost * 1000).toFixed(2)}m`; // millidollars
  }
  return `$${cost.toFixed(4)}`;
}

/**
 * Format tokens for display
 */
export function formatTokens(tokens: number): string {
  if (tokens > 1000000) {
    return `${(tokens / 1000000).toFixed(1)}M tokens`;
  }
  if (tokens > 1000) {
    return `${(tokens / 1000).toFixed(1)}K tokens`;
  }
  return `${tokens} tokens`;
}

/**
 * Create a human-readable savings message
 */
export function createSavingsMessage(metrics: CostMetrics): string {
  const { totalTokensSaved, totalCostSaved } = metrics;

  if (totalCostSaved < 0.001) {
    return `💰 Saved ${formatTokens(totalTokensSaved)} (${formatCost(totalCostSaved)})`;
  }

  return `💰 Saved ${formatTokens(totalTokensSaved)} (${formatCost(totalCostSaved)})`;
}

/**
 * Create a monthly/yearly cost report
 */
export function createCostReport(metrics: CostMetrics): string {
  const lines = [
    "📊 COST TRACKING REPORT",
    "─".repeat(50),
    "",
    "💵 MONTHLY METRICS",
    `  Current Month: ${formatCost(metrics.monthlyCurrent || 0)} spent`,
    `  Projected:    ${formatCost(metrics.monthlyProjected || 0)} (end of month)`,
    "",
    "📈 YEARLY METRICS",
    `  Current Year:   ${formatCost(metrics.yearly?.cost || 0)} spent`,
    `  Yearly Trend:   ${formatCost(metrics.yearlyProjected || 0)} (annualized)`,
    "",
    "📉 TOTAL SAVINGS",
    `  All-time:       ${formatCost(metrics.totalCostSaved)}`,
    `  Tokens Saved:   ${formatTokens(metrics.totalTokensSaved)}`,
  ];

  return lines.join("\n");
}

/**
 * Get monthly cost breakdown
 */
export function getMonthlyBreakdown(metrics: CostMetrics): Array<{ month: string; cost: number }> {
  return (metrics.monthly || [])
    .sort((a, b) => a.month.localeCompare(b.month))
    .map(({ month, cost }) => ({ month, cost }));
}

/**
 * Get daily cost breakdown
 */
export function getDailyBreakdown(metrics: CostMetrics): Array<{ date: string; cost: number }> {
  return (metrics.daily || [])
    .sort((a, b) => a.date.localeCompare(b.date))
    .slice(-30) // Last 30 days
    .map(({ date, cost }) => ({ date, cost }));
}

export default {
  CostTracker,
  formatCost,
  formatTokens,
  createSavingsMessage,
  createCostReport,
  getMonthlyBreakdown,
  getDailyBreakdown,
};
