import type { PendingInvitation } from "$lib/types";
import { timeAsync } from "$lib/utils/timing";
import { BaseRepository, handleError } from "./base";

export class InvitationRepository extends BaseRepository {
  async createInvitation(
    from: string,
    to: string,
    workspace_id: string,
    role_id: string,
  ): Promise<PendingInvitation> {
    return timeAsync(
      "db.createInvitation",
      async () => {
        const { data, error } = await this.client
          .from("workspace_invitations")
          .insert({
            from: from,
            to: to,
            workspace_id: workspace_id,
            role_id: role_id,
            status: "PENDING",
          })
          .select()
          .single();

        if (error) {
          handleError(error, "Failed to create invitation");
        }
        if (!data) {
          throw new Error("unknown error");
        }

        return {
          id: data.id,
          from: data.from,
          to: data.to,
          workspaceId: data.workspace_id,
          roleId: data.role_id,
          status: data.status,
          createdAt: new Date(data.created_at),
        };
      },
      {
        from: from,
        to: to,
      },
    );
  }

  async markInvitationAsAccepted(id: string): Promise<void> {
    return timeAsync(
      "db.markInvitationMarked",
      async () => {
        const { error } = await this.client
          .from("workspace_invitations")
          .update({ status: "accepted" })
          .eq("id", id);

        if (error) {
          handleError(error, "failed to update invitation");
        }
      },
      { id },
    );
  }

  async getPendingInvitationsFrom(
    userId: string,
    status: string,
  ): Promise<PendingInvitation[]> {
    return timeAsync(
      "db.getPendingInvitationsFrom",
      async () => {
        const { data, error } = await this.client
          .from("workspace_invitations")
          .select("*")
          .eq("from", userId)
          .eq("status", status);

        handleError(error, "Failed to get pending invitations");
        if (!data) {
          return [];
        }

        return data.map((item) => ({
          id: item.id,
          from: item.from,
          to: item.to,
          workspaceId: item.workspace_id,
          roleId: item.role_id,
          createdAt: new Date(item.created_at),
          status: item.status,
        }));
      },
      { userId },
    );
  }

  async getPendingInvitationsTo(
    userId: string,
    status: string,
  ): Promise<PendingInvitation[]> {
    return timeAsync(
      "db.getPendingInvitationsTo",
      async () => {
        const { data, error } = await this.client
          .from("workspace_invitations")
          .select("*")
          .eq("to", userId)
          .eq("status", status);

        handleError(error, "Failed to get pending invitations");
        if (!data) {
          return [];
        }

        return data.map((item) => ({
          id: item.id,
          from: item.from,
          to: item.to,
          workspaceId: item.workspace_id,
          roleId: item.role_id,
          createdAt: new Date(item.created_at),
          status: item.status,
        }));
      },
      { userId },
    );
  }
}