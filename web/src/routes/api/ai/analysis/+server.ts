import { getExperimentAndMetrics } from "$lib/server/database";
import type { ExperimentAndMetrics } from "$lib/types";
import { createAnthropicClient } from "$lib/server/llm";
import { AnalysisSchema } from "./schema";
import zodToJsonSchema from "zod-to-json-schema";

const MODEL = "claude-3-7-sonnet-20250219";

export async function GET({ url }: { url: URL }) {
  const experimentId = url.searchParams.get("experimentId");
  if (!experimentId) {
    return new Response(JSON.stringify({ error: "experimentId is required" }), {
      status: 400,
    });
  }

  const data = (await getExperimentAndMetrics(
    experimentId,
  )) as ExperimentAndMetrics;
  const system = createSystemPrompt();
  const user = createUserPrompt(data);
  try {
    const client = createAnthropicClient();
    const msg = await client.messages.create({
      model: MODEL,
      system,
      messages: [{ role: "user", content: user }],
      max_tokens: 21_333,
      temperature: 0.6,
    });
    const textItem = msg.content.find((item) => item.type === "text");
    const raw = textItem
      ? textItem.text
      : JSON.stringify({ error: "invalid model response" });
    const parsed = parseOutput(raw);
    return new Response(JSON.stringify(parsed), {
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    console.error("LLM analysis error", err);
    return new Response(
      JSON.stringify({ error: "structured analysis failed" }),
      { status: 500 },
    );
  }
}

function createSystemPrompt(): string {
  const schema = zodToJsonSchema(AnalysisSchema, "AnalysisSchema");
  return `
# You are a ML lead tasked to analyze and compile a report of machine learning experiments for stakeholders and model trainers.

Respond using the following JSON schema:

${JSON.stringify(schema, null, 2)}

## Notes
* Only recommend changes to hyperparameters that were logged.
`.trim();
}

function createUserPrompt({
  experiment,
  metrics,
}: ExperimentAndMetrics): string {
  const lines: string[] = [
    "# Experiment Analysis Request",
    `Analyze experiment: ${experiment.name} (ID: ${experiment.id})`,
  ];

  if (experiment.description) {
    lines.push("## Description", experiment.description);
  }
  if (experiment.tags?.length) {
    lines.push("## Tags", experiment.tags.join(", "));
  }
  if (experiment.hyperparams?.length) {
    lines.push("## Hyperparameters");
    experiment.hyperparams.forEach(({ key, value }) => {
      lines.push(`- ${key}: ${value}`);
    });
  }

  lines.push("## Metrics Summary");
  if (!metrics.length) {
    lines.push("No metrics recorded.");
  } else {
    const grouped = metrics.reduce<Record<string, typeof metrics>>((acc, m) => {
      (acc[m.name] ||= []).push(m);
      return acc;
    }, {});

    Object.entries(grouped).forEach(([name, entries]) => {
      lines.push(`\n### ${name}`);
      const sortedEntries = [...entries].sort((a, b) => {
        if (a.step !== undefined && b.step !== undefined) {
          return a.step - b.step;
        }
        return (
          new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
        );
      });

      const hasSteps = sortedEntries.some((e) => e.step !== undefined);
      if (hasSteps) {
        lines.push("| Step | Value |");
        lines.push("| ---- | ----- |");
        sortedEntries.forEach((m) => {
          lines.push(`| ${m.step ?? "N/A"} | ${m.value} |`);
        });
      } else {
        lines.push(`Values: ${sortedEntries.map((m) => m.value).join(", ")}`);
        lines.push(`Latest value: ${sortedEntries.at(-1)?.value ?? "N/A"}`);
      }

      const metricWithMetadata = sortedEntries.find(
        (m) => m.metadata && Object.keys(m.metadata).length > 0,
      );
      if (metricWithMetadata?.metadata) {
        lines.push("Metadata example:");
        lines.push("```");
        lines.push(JSON.stringify(metricWithMetadata.metadata, null, 2));
        lines.push("```");
      }
    });

    lines.push("\n## Summary Statistics");
    Object.entries(grouped).forEach(([name, entries]) => {
      const values = entries.map((m) => m.value);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const avg = values.reduce((sum, v) => sum + v, 0) / values.length;

      lines.push(`\n### ${name}`);
      lines.push(`- Count: ${values.length}`);
      lines.push(`- Min: ${min}`);
      lines.push(`- Max: ${max}`);
      lines.push(`- Average: ${avg.toFixed(4)}`);

      if (values.length > 1) {
        const first = values[0];
        const last = values.at(-1)!;
        const change = last - first;
        const percentChange = (change / Math.abs(first)) * 100;

        lines.push(
          `- Change: ${change > 0 ? "+" : ""}${change.toFixed(4)} (${percentChange > 0 ? "+" : ""}${percentChange.toFixed(2)}%)`,
        );
        lines.push(
          `- Trend: ${change > 0 ? "Increasing" : change < 0 ? "Decreasing" : "Stable"}`,
        );
      }
    });
  }

  lines.push(
    "## Analysis Request",
    "Please analyze this experiment data and provide insights on performance, trends, and improvements.",
  );

  return lines.join("\n\n");
}

function parseOutput(raw: string) {
  let jsonText = raw.trim();
  if (jsonText.startsWith("```json")) {
    jsonText = jsonText.slice(7).trim();
  } else if (jsonText.startsWith("```")) {
    jsonText = jsonText.slice(3).trim();
  }
  if (jsonText.endsWith("```")) {
    jsonText = jsonText.slice(0, -3).trim();
  }

  try {
    return JSON.parse(jsonText);
  } catch (e) {
    console.warn(`Failed to parse JSON output: ${e}`);
    throw e;
  }
}
