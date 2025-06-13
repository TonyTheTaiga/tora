import { PRIVATE_ANTHROPIC_KEY } from "$env/static/private";
import Anthropic from "@anthropic-ai/sdk";

export function createAnthropicClient() {
  const client = new Anthropic({
    apiKey: PRIVATE_ANTHROPIC_KEY,
  });
  return client;
}
