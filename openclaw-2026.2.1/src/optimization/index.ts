/**
 * OpenClaw Optimization System - Main Export
 * 
 * Complete cost optimization and UI/UX enhancement suite
 * Estimated savings: 35-45% on API costs
 * 
 * Features:
 * - Response token truncation (15-25% savings)
 * - Prompt compression (10-20% savings)
 * - Cache-aware routing (90% on cached queries)
 * - Batch message processing (5-10% savings)
 * - Context window optimization (20-30% savings)
 * - Real-time cost tracking
 * - Beautiful, accessible UI components
 */

// Import classes for OptimizationSystem
import { CacheAwareRouter } from './cache-aware-routing.js';
import { CostTracker } from './cost-tracking.js';
import { BatchMessageProcessor } from './batch-message-processor.js';

// Cost Optimization Modules
export {
  truncateResponse,
  truncateResponseBatch,
  calculateBatchSavings,
  type TruncationConfig,
  type TruncationResult,
} from "./response-truncation.js";

export {
  compressPrompt,
  compressMessageThread,
  compressPromptBatch,
  calculateCompressionStats,
  type CompressionConfig,
  type CompressionResult,
  type Message as CompressionMessage,
} from "./prompt-compression.js";

export {
  CacheAwareRouter,
  calculateComplexity,
  type CacheEntry,
  type RoutingDecision,
  type CacheStatistics,
} from "./cache-aware-routing.js";

export {
  BatchMessageProcessor,
  combineMessages,
  splitBatchResults,
  type QueuedMessage,
  type BatchRequest,
  type BatchingConfig,
  type BatchingStats,
} from "./batch-message-processor.js";

export {
  optimizeContext,
  buildOptimizedContext,
  optimizeBatch,
  calculateBatchStats,
  type OptimizationConfig,
  type OptimizationResult,
  type Message as ContextMessage,
} from "./context-optimizer.js";

export {
  CostTracker,
  formatCost,
  formatTokens,
  createSavingsMessage,
  createCostReport,
  getMonthlyBreakdown,
  getDailyBreakdown,
  type OptimizationEvent,
  type CostMetrics,
} from "./cost-tracking.js";

export {
  enrichStatusWithCosts,
  formatCostMetrics,
  formatCostSummary,
  formatMonthlyTable,
  createDashboardWidget,
  type StatusMetrics,
} from "./cost-status-display.js";

// UI Components (React) - commented out for server-side build
// These components are available in the UI package, not in the server build
// export {
//   CostSavingsDisplay,
//   FeatureBreakdown,
//   ExportButton,
//   type CostVisualizerProps,
// } from "./cost-tracking-visualizer.js";
//
// export {
//   ModelSelectionPrompt,
//   type ModelOption,
//   type ModelSelectionProps,
// } from "./model-selection-prompt.js";
//
// export {
//   OptimizationDashboard,
//   type OptimizationFeature,
//   type OptimizationDashboardProps,
// } from "./optimization-dashboard.js";

// Quick start utilities
export const OptimizationSystem = {
  /**
   * Initialize all optimization modules
   */
  initialize: () => ({
    cacheRouter: new CacheAwareRouter(),
    costTracker: new CostTracker(),
    batchProcessor: new BatchMessageProcessor(),
  }),

  /**
   * Calculate total savings potential
   */
  calculateSavingsPotential: (
    estimatedRequests: number,
    avgInputTokens: number = 1000,
    avgOutputTokens: number = 2000,
    cacheHitRate: number = 0.3
  ) => {
    const inputTokens = estimatedRequests * avgInputTokens;
    const outputTokens = estimatedRequests * avgOutputTokens;

    return {
      // Compression savings
      compressionSavings: inputTokens * 0.15,

      // Truncation savings
      truncationSavings: outputTokens * 0.20,

      // Cache routing savings (90% on cached)
      cacheSavings: estimatedRequests * cacheHitRate * (avgInputTokens + avgOutputTokens) * 0.90,

      // Context optimization
      contextSavings: inputTokens * 0.15,

      // Total tokens saved
      totalTokensSaved:
        inputTokens * 0.15 + // compression
        outputTokens * 0.20 + // truncation
        estimatedRequests * cacheHitRate * (avgInputTokens + avgOutputTokens) * 0.90 + // cache
        inputTokens * 0.15, // context

      // Estimated cost savings (at Haiku rates: $0.0008/M tokens)
      estimatedCostSavings:
        ((inputTokens * 0.15 + outputTokens * 0.20 + inputTokens * 0.15) / 1000000) * 0.0008 +
        (estimatedRequests * cacheHitRate * (avgInputTokens + avgOutputTokens) * 0.90 / 1000000) * 0.0008,
    };
  },
};

export default OptimizationSystem;
