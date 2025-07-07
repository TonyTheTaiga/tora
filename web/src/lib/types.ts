export interface Experiment {
  id: string;
  name: string;
  description: string;
  hyperparams: HyperParam[];
  tags: string[];
  createdAt: Date;
  updatedAt: Date;
  availableMetrics: string[];
  workspaceId?: string;
}

export interface Metric {
  id: number;
  experiment_id: string;
  name: string;
  value: number;
  step?: number;
  metadata?: any;
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

export type WorkspaceRole = "VIEWER" | "EDITOR" | "ADMIN" | "OWNER";

export interface Workspace {
  id: string;
  name: string;
  description: string | null;
  createdAt: Date;
  role: string;
}

export interface PendingInvitation {
  id: string;
  from: string;
  to: string;
  workspaceId: string;
  roleId: string;
  createdAt: Date;
}

export function isWorkspace(obj: unknown): obj is Workspace {
  if (typeof obj !== "object" || obj === null) return false;
  const w = obj as Record<string, unknown>;
  return (
    typeof w.id === "string" &&
    typeof w.role === "string" &&
    typeof w.name === "string" &&
    (typeof w.description === "string" || w.description === null) &&
    typeof w.createdAt === "string"
  );
}

export interface ApiKey {
  id: string;
  key?: string;
  name: string;
  createdAt: Date;
  lastUsed: Date;
  revoked: boolean;
}