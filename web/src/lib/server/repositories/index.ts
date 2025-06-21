import type { SupabaseClient } from "@supabase/supabase-js";
import type { Database } from "../database.types";
import { ExperimentRepository } from "./experiment";
import { MetricRepository } from "./metric";
import { WorkspaceRepository } from "./workspace";
import { ApiKeyRepository } from "./api-key";
import { InvitationRepository } from "./invitation";

export class RepositoryFactory {
  private client: SupabaseClient<Database>;
  private experimentRepo?: ExperimentRepository;
  private metricRepo?: MetricRepository;
  private workspaceRepo?: WorkspaceRepository;
  private apiKeyRepo?: ApiKeyRepository;
  private invitationRepo?: InvitationRepository;

  constructor(client: SupabaseClient<Database>) {
    this.client = client;
  }

  get experiments(): ExperimentRepository {
    if (!this.experimentRepo) {
      this.experimentRepo = new ExperimentRepository(this.client);
    }
    return this.experimentRepo;
  }

  get metrics(): MetricRepository {
    if (!this.metricRepo) {
      this.metricRepo = new MetricRepository(this.client);
    }
    return this.metricRepo;
  }

  get workspaces(): WorkspaceRepository {
    if (!this.workspaceRepo) {
      this.workspaceRepo = new WorkspaceRepository(this.client);
    }
    return this.workspaceRepo;
  }

  get apiKeys(): ApiKeyRepository {
    if (!this.apiKeyRepo) {
      this.apiKeyRepo = new ApiKeyRepository(this.client);
    }
    return this.apiKeyRepo;
  }

  get invitations(): InvitationRepository {
    if (!this.invitationRepo) {
      this.invitationRepo = new InvitationRepository(this.client);
    }
    return this.invitationRepo;
  }
}

export function createRepositories(client: SupabaseClient<Database>): RepositoryFactory {
  return new RepositoryFactory(client);
}

export * from "./experiment";
export * from "./metric";
export * from "./workspace";
export * from "./api-key";
export * from "./invitation";
export * from "./base";