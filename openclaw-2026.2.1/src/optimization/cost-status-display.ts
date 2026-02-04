/**
 * Cost Status Display
 * Formats cost metrics for CLI and dashboard status output
 */

import { CostTracker, CostMetrics, formatCost, formatTokens } from "./cost-tracking.js";

export interface StatusMetrics {
  model: string;
  tokens: string;
  costCurrent: string;
  costMonthly: string;
  costYearly: string;
  savingsSession: string;
  savingsMonthly: string;
  savingsYearly: string;
}

/**
 * Format status line with cost metrics
 * Input: "connected | idle agent main | session main (webchat:...) | anthropic/claude-haiku-4-5-20251001 | tokens 52k/200k (26%)"
 * Output: Same + cost metrics
 */
export function enrichStatusWithCosts(
  statusLine: string,
  metrics: CostMetrics
): string {
  const parts = statusLine.split(" | ");

  // Add cost metrics at the end
  const monthlyCost = formatCost(metrics.monthlyCurrent || 0);
  const monthlyProjected = formatCost(metrics.monthlyProjected || 0);
  const yearlyProjected = formatCost(metrics.yearlyProjected || 0);
  const sessionSavings = formatCost(metrics.totalCostSaved);

  // Build the enhanced status
  const enhanced = [
    ...parts,
    `cost $${(metrics.monthlyCurrent || 0).toFixed(2)}/mo (projected: $${(metrics.monthlyProjected || 0).toFixed(2)})`,
    `savings $${(metrics.totalCostSaved).toFixed(2)} this session`,
  ];

  return enhanced.join(" | ");
}

/**
 * Create a formatted cost metrics block for display
 */
export function formatCostMetrics(metrics: CostMetrics): string {
  const lines: string[] = [];

  lines.push(""); // Blank line for spacing
  lines.push("╔════════════════════════════════════════════════════════════╗");
  lines.push("║                    💰 COST METRICS                         ║");
  lines.push("╠════════════════════════════════════════════════════════════╣");

  // Current session
  lines.push("║ SESSION                                                    ║");
  lines.push(
    `║   Savings: ${formatCost(metrics.totalCostSaved).padEnd(48)} ║`
  );
  lines.push(
    `║   Tokens:  ${formatTokens(metrics.totalTokensSaved).padEnd(48)} ║`
  );

  lines.push("║                                                            ║");
  lines.push("║ THIS MONTH                                                 ║");
  const currentMonth = formatCost(metrics.monthlyCurrent || 0);
  const projectedMonth = formatCost(metrics.monthlyProjected || 0);
  lines.push(`║   Spent:     ${currentMonth.padEnd(49)} ║`);
  lines.push(`║   Projected: ${projectedMonth.padEnd(49)} ║`);

  lines.push("║                                                            ║");
  lines.push("║ ANNUAL PROJECTION                                          ║");
  const yearlyProj = formatCost(metrics.yearlyProjected || 0);
  lines.push(`║   ${yearlyProj.padEnd(53)} ║`);

  lines.push("║                                                            ║");
  lines.push("║ BY FEATURE                                                 ║");

  // Top 5 features by savings
  const features = Object.entries(metrics.byFeature || {})
    .sort((a, b) => b[1].costSaved - a[1].costSaved)
    .slice(0, 5);

  for (const [name, data] of features) {
    const saved = formatCost(data.costSaved);
    lines.push(`║   ${name.padEnd(20)} ${saved.padEnd(29)} ║`);
  }

  lines.push("╚════════════════════════════════════════════════════════════╝");
  lines.push("");

  return lines.join("\n");
}

/**
 * Create a compact single-line cost summary
 */
export function formatCostSummary(metrics: CostMetrics): string {
  const monthly = `$${(metrics.monthlyCurrent || 0).toFixed(2)}`;
  const projected = `$${(metrics.monthlyProjected || 0).toFixed(2)}`;
  const yearly = `$${(metrics.yearlyProjected || 0).toFixed(2)}/yr`;
  const saved = formatCost(metrics.totalCostSaved);

  return `💰 ${monthly}/mo | 📈 ${projected} proj | 📊 ${yearly} | ✨ ${saved} saved`;
}

/**
 * Create a detailed table with monthly history
 */
export function formatMonthlyTable(metrics: CostMetrics): string {
  const monthly = metrics.monthly || [];
  if (monthly.length === 0) {
    return "No monthly data available yet.";
  }

  const lines: string[] = [];
  lines.push("MONTHLY BREAKDOWN");
  lines.push("────────────────────────────────");
  lines.push("Month      | Saved        | Tokens");
  lines.push("─".repeat(33));

  // Sort by month descending (newest first)
  const sorted = [...monthly].sort((a, b) => b.month.localeCompare(a.month));

  for (const entry of sorted.slice(0, 12)) {
    // Last 12 months
    const month = entry.month;
    const cost = formatCost(entry.cost);
    const tokens = formatTokens(entry.tokens);
    lines.push(`${month} | ${cost.padEnd(12)} | ${tokens}`);
  }

  lines.push("");
  return lines.join("\n");
}

/**
 * Create a dashboard widget showing key metrics
 */
export function createDashboardWidget(metrics: CostMetrics): {
  title: string;
  sections: Array<{ label: string; value: string; emoji: string }>;
} {
  return {
    title: "Cost Metrics",
    sections: [
      {
        label: "This Month",
        value: formatCost(metrics.monthlyCurrent || 0),
        emoji: "💵",
      },
      {
        label: "Monthly Projection",
        value: formatCost(metrics.monthlyProjected || 0),
        emoji: "📈",
      },
      {
        label: "Annual Projection",
        value: formatCost(metrics.yearlyProjected || 0),
        emoji: "📊",
      },
      {
        label: "Savings (Session)",
        value: formatCost(metrics.totalCostSaved),
        emoji: "✨",
      },
      {
        label: "Tokens Saved",
        value: formatTokens(metrics.totalTokensSaved),
        emoji: "🎯",
      },
    ],
  };
}

/**
 * Export all functions
 */
export default {
  enrichStatusWithCosts,
  formatCostMetrics,
  formatCostSummary,
  formatMonthlyTable,
  createDashboardWidget,
};
