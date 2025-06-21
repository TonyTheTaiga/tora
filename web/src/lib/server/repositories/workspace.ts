import type { Workspace, Experiment, HyperParam } from "$lib/types";
import { timeAsync } from "$lib/utils/timing";
import { BaseRepository, handleError, mapToWorkspace } from "./base";

export class WorkspaceRepository extends BaseRepository {
  async getWorkspacesV2(
    userId: string,
    roles: string[],
  ): Promise<Workspace[]> {
    const { data, error } = await this.client
      .from("workspace")
      .select(
        `
        id,
        name,
        description,
        created_at,
        user_workspaces!inner (
          user_id,
          workspace_role (
            name
          )
        )
      `,
      )
      .eq("user_workspaces.user_id", userId)
      .in("user_workspaces.workspace_role.name", roles);

    handleError(error, "Failed to get workspaces");
    return data?.map(mapToWorkspace) ?? [];
  }

  async getWorkspacesAndExperiments(
    userId: string,
    roles: string[],
  ): Promise<{
    workspaces: Workspace[];
    experiments: Experiment[];
  }> {
    return timeAsync(
      "db.getWorkspacesAndExperiments",
      async () => {
        // Get workspaces for the user
        const { data: workspaceData, error: workspaceError } = await this.client
          .from("workspace")
          .select(
            `
            id,
            name,
            description,
            created_at,
            user_workspaces!inner (
              user_id,
              workspace_role (
                name
              )
            )
          `,
          )
          .eq("user_workspaces.user_id", userId)
          .in("user_workspaces.workspace_role.name", roles);

        handleError(workspaceError, "Failed to get workspaces");
        const workspaces = workspaceData?.map(mapToWorkspace) ?? [];

        if (workspaces.length === 0) {
          return { workspaces: [], experiments: [] };
        }

        const workspaceIds = workspaces.map((w) => w.id);
        const { data: experimentData, error: experimentError } = await this.client
          .from("workspace_experiments")
          .select(
            `
            workspace_id,
            experiment:experiment_id (
              id,
              name,
              description,
              hyperparams,
              tags,
              created_at,
              updated_at
            )
          `,
          )
          .in("workspace_id", workspaceIds);

        handleError(experimentError, "Failed to get experiments");

        const experiments: Experiment[] = [];
        const seenExperimentIds = new Set<string>();
        experimentData?.forEach((item) => {
          if (item.experiment && !seenExperimentIds.has(item.experiment.id)) {
            seenExperimentIds.add(item.experiment.id);
            experiments.push({
              id: item.experiment.id,
              name: item.experiment.name,
              description: item.experiment.description ?? "",
              hyperparams:
                (item.experiment.hyperparams as unknown as HyperParam[]) ??
                [],
              tags: item.experiment.tags ?? [],
              createdAt: new Date(item.experiment.created_at),
              updatedAt: new Date(item.experiment.updated_at),
              availableMetrics: [],
              workspaceId: item.workspace_id,
            });
          }
        });

        return { workspaces, experiments };
      },
      { userId },
    );
  }

  async createWorkspace(
    name: string,
    description: string | null,
    userId: string,
  ): Promise<Workspace> {
    const { data: workspaceData, error: workspaceError } = await this.client
      .from("workspace")
      .insert({ name, description })
      .select()
      .single();

    handleError(workspaceError, "Failed to create workspace");
    if (!workspaceData)
      throw new Error("Workspace creation returned no data.");

    const { data: ownerRole, error: roleError } = await this.client
      .from("workspace_role")
      .select("id")
      .eq("name", "OWNER")
      .single();

    handleError(roleError, "Failed to get OWNER role");
    if (!ownerRole) throw new Error("OWNER role not found");

    const { error: userWorkspaceError } = await this.client
      .from("user_workspaces")
      .insert({
        user_id: userId,
        workspace_id: workspaceData.id,
        role_id: ownerRole.id,
      });

    handleError(userWorkspaceError, "Failed to add user to workspace");

    const workspaceWithRole = {
      ...workspaceData,
      user_workspaces: [{ workspace_role: { name: "OWNER" } }],
    };

    return mapToWorkspace(workspaceWithRole);
  }

  async deleteWorkspace(id: string): Promise<void> {
    const { error } = await this.client.from("workspace").delete().eq("id", id);
    handleError(error, "Failed to delete workspace");
  }

  async removeWorkspaceRole(workspaceID: string, userId: string): Promise<void> {
    const { error } = await this.client
      .from("user_workspaces")
      .delete()
      .eq("user_id", userId)
      .eq("workspace_id", workspaceID);
    handleError(error, "Failed to remove workspace role");
  }
}