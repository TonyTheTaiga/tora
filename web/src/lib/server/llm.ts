import { env } from "$env/dynamic/private";
import Anthropic from "@anthropic-ai/sdk";

export function createAnthropicClient() {
  const client = new Anthropic({
    apiKey: env.PRIVATE_ANTHROPIC_KEY,
  });
  return client;
}
