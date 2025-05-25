import type { Json } from "./server/database.types";

export interface ExperimentAndMetrics {
  experiment: Experiment;
  metrics: Metric[];
}

export type Visibility = "PUBLIC" | "PRIVATE";

export type ExperimentStatus = "COMPLETED" | "RUNNING" | "FAILED" | "DRAFT" | "OTHER";

export interface Experiment {
  id: string;
  user_id?: string;
  name: string;
  description?: string | null;
  availableMetrics: string[];
  hyperparams?: HyperParam[] | null;
  tags?: string[] | null;
  createdAt: Date; // Keep existing fields
  updatedAt?: Date; // Optional: consider adding this if you want to sort by last updated
  visibility?: Visibility;
  status?: ExperimentStatus; // New field for experiment status
  keyMetrics?: Array<{ name: string; value: string | number }>; // New field for key metrics
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

export interface Workspace {
  id: string;
  user_id: string;
  name: string;
  description: string | null;
  created_at: Date;
}
