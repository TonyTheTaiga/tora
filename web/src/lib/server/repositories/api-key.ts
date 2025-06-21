import type { ApiKey } from "$lib/types";
import { BaseRepository, handleError } from "./base";

export class ApiKeyRepository extends BaseRepository {
  async getApiKeys(userId: string): Promise<ApiKey[]> {
    const { data, error } = await this.client
      .from("api_keys")
      .select("id, name, created_at, last_used, revoked")
      .eq("user_id", userId)
      .eq("revoked", false)
      .order("created_at", { ascending: false });

    handleError(error, "Failed to get API keys");
    return (
      data?.map((row) => ({
        id: row.id,
        name: row.name,
        createdAt: new Date(row.created_at),
        lastUsed: new Date(row.last_used),
        revoked: row.revoked,
      })) ?? []
    );
  }

  async createApiKey(
    userId: string,
    name: string,
    keyHash: string,
  ): Promise<ApiKey> {
    const { data, error } = await this.client
      .from("api_keys")
      .insert({
        key_hash: keyHash,
        name,
        user_id: userId,
        revoked: false,
      })
      .select()
      .single();

    handleError(error, "Failed to create API key");
    if (!data) throw new Error("API key creation returned no data.");
    return {
      id: data.id,
      name: data.name,
      revoked: data.revoked,
      createdAt: new Date(data.created_at),
      lastUsed: new Date(data.last_used),
    };
  }

  async revokeApiKey(userId: string, keyId: string): Promise<void> {
    const { data: keyData, error: fetchError } = await this.client
      .from("api_keys")
      .select("id")
      .eq("id", keyId)
      .eq("user_id", userId)
      .single();

    if (fetchError || !keyData) {
      throw new Error("API key not found");
    }

    const { error: updateError } = await this.client
      .from("api_keys")
      .update({ revoked: true })
      .eq("id", keyId)
      .eq("user_id", userId);

    handleError(updateError, "Failed to revoke API key");
  }

  async lookupApiKey(keyHash: string): Promise<{ user_id: string } | null> {
    const { data, error } = await this.client
      .from("api_keys")
      .select("user_id")
      .eq("key_hash", keyHash)
      .eq("revoked", false)
      .single();

    if (error || !data?.user_id) {
      if (error) console.error("API key lookup error:", error.message);
      return null;
    }

    return { user_id: data.user_id };
  }

  async updateApiKeyLastUsed(keyHash: string): Promise<void> {
    const { error } = await this.client
      .from("api_keys")
      .update({ last_used: new Date().toISOString() })
      .eq("key_hash", keyHash)
      .eq("revoked", false);

    if (error) {
      console.warn("Failed to update API key last_used:", error.message);
    }
  }
}