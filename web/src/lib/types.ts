import type { Json } from "./server/database.types";

export interface ExperimentAndMetrics {
  experiment: Experiment;
  metrics: Metric[];
}

export type Visibility = "PUBLIC" | "PRIVATE";

export interface Experiment {
  id: string;
  user_id: string;
  name: string;
  description?: string | null;
  availableMetrics?: string[] | null;
  hyperparams?: HyperParam[] | null;
  tags?: string[] | null;
  createdAt: Date;
  visibility?: Visibility;
}

export interface Metric {
  id: number;
  experiment_id: string;
  name: string;
  value: number;
  step?: number;
  metadata?: Json;
  created_at: string;
}

export interface HyperParam {
  key: string;
  value: string | number;
}

export interface HPRecommendation {
  recommendation: string;
  level: number;
}

export interface ExperimentAnalysis {
  summary: string;
  insights: string[];
  recommendations: string[];
  hyperparameter_recommendations: Record<string, HPRecommendation>;
}
