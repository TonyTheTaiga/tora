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

export interface ApiKey {
  id: string;
  key?: string;
  name: string;
  createdAt: Date;
  lastUsed: Date;
  revoked: boolean;
}

export interface SessionData {
  access_token: string;
  expires_in: number;
  expires_at: number;
  refresh_token: string;
  user: {
    id: string;
    email: string;
  };
}
