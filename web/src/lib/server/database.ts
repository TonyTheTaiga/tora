import type { SupabaseClient } from "@supabase/supabase-js";
import type { Database } from "./database.types";
import { createRepositories } from "./repositories";

export function createDbClient(client: SupabaseClient<Database>) {
  const repositories = createRepositories(client);

  return {
    // --- Experiment Methods ---
    createExperiment: repositories.experiments.createExperiment.bind(
      repositories.experiments,
    ),
    getExperiments: repositories.experiments.getExperiments.bind(
      repositories.experiments,
    ),
    getExperiment: repositories.experiments.getExperiment.bind(
      repositories.experiments,
    ),
    getPublicExperiments: repositories.experiments.getPublicExperiments.bind(
      repositories.experiments,
    ),
    checkExperimentAccess: repositories.experiments.checkExperimentAccess.bind(
      repositories.experiments,
    ),
    updateExperiment: repositories.experiments.updateExperiment.bind(
      repositories.experiments,
    ),
    deleteExperiment: repositories.experiments.deleteExperiment.bind(
      repositories.experiments,
    ),

    // --- Metric Methods ---
    getMetrics: repositories.metrics.getMetrics.bind(repositories.metrics),
    createMetric: repositories.metrics.createMetric.bind(repositories.metrics),
    batchCreateMetric: repositories.metrics.batchCreateMetric.bind(
      repositories.metrics,
    ),

    // --- Workspace Methods ---
    getWorkspacesV2: repositories.workspaces.getWorkspacesV2.bind(
      repositories.workspaces,
    ),
    getWorkspacesAndExperiments:
      repositories.workspaces.getWorkspacesAndExperiments.bind(
        repositories.workspaces,
      ),
    createWorkspace: repositories.workspaces.createWorkspace.bind(
      repositories.workspaces,
    ),
    deleteWorkspace: repositories.workspaces.deleteWorkspace.bind(
      repositories.workspaces,
    ),
    removeWorkspaceRole: repositories.workspaces.removeWorkspaceRole.bind(
      repositories.workspaces,
    ),

    addExperimentToWorkspace:
      repositories.workspaces.addExperimentToWorkspace.bind(
        repositories.workspaces,
      ),

    // --- API Key Methods ---
    getApiKeys: repositories.apiKeys.getApiKeys.bind(repositories.apiKeys),
    createApiKey: repositories.apiKeys.createApiKey.bind(repositories.apiKeys),
    revokeApiKey: repositories.apiKeys.revokeApiKey.bind(repositories.apiKeys),
    lookupApiKey: repositories.apiKeys.lookupApiKey.bind(repositories.apiKeys),
    updateApiKeyLastUsed: repositories.apiKeys.updateApiKeyLastUsed.bind(
      repositories.apiKeys,
    ),

    // --- Experiment and Metrics Combined Methods ---
    getExperimentsAndMetrics:
      repositories.experiments.getExperimentsAndMetrics.bind(
        repositories.experiments,
      ),

    // --- Workspace Invitations ---
    createInvitation: repositories.invitations.createInvitation.bind(
      repositories.invitations,
    ),
    markInvitationAsAccepted:
      repositories.invitations.markInvitationAsAccepted.bind(
        repositories.invitations,
      ),
    getPendingInvitationsFrom:
      repositories.invitations.getPendingInvitationsFrom.bind(
        repositories.invitations,
      ),
    getPendingInvitationsTo:
      repositories.invitations.getPendingInvitationsTo.bind(
        repositories.invitations,
      ),
  };
}
