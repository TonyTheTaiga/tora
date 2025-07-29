<script lang="ts">
  import type { Experiment, Workspace } from "$lib/types";

  let workspaces = $state<Workspace[]>([]);
  let loading = $state({
    workspaces: true,
    experiments: false,
    experimentDetails: false,
  });
  let errors = $state({
    workspaces: null as string | null,
    experiments: null as string | null,
    experimentDetails: null as string | null,
  });

  let selectedWorkspace = $state<Workspace | null>(null);
  let selectedExperiment = $state<Experiment | null>(null);
  let experiments = $state<Experiment[]>([]);
  let scalarMetrics = $state<any[]>([]);

  let workspaceSearchQuery = $state("");
  let experimentSearchQuery = $state("");

  async function loadWorkspaces() {
    try {
      loading.workspaces = true;
      errors.workspaces = null;
      const response = await fetch("/api/dashboard/overview");
      if (!response.ok)
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      const apiResponse = await response.json();
      const data = apiResponse.data;
      if (!data || !data.workspaces)
        throw new Error("Invalid response structure from dashboard API");

      workspaces = data.workspaces.map((ws: any) => ({
        id: ws.id,
        name: ws.name,
        description: ws.description,
        createdAt: new Date(ws.created_at),
        role: ws.role,
        experimentCount: ws.experiment_count,
      }));
      if (workspaces.length > 0 && !selectedWorkspace) {
        selectedWorkspace = workspaces[0];
      }
    } catch (error) {
      console.error("Failed to load workspaces:", error);
      errors.workspaces =
        error instanceof Error ? error.message : "Failed to load workspaces";
    } finally {
      loading.workspaces = false;
    }
  }

  async function loadExperiments(workspaceId: string) {
    try {
      loading.experiments = true;
      errors.experiments = null;
      const response = await fetch(
        `/api/workspaces/${workspaceId}/experiments`,
      );
      if (!response.ok)
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      const apiResponse = await response.json();
      const data = apiResponse.data;
      if (!data || !Array.isArray(data))
        throw new Error("Invalid response structure from experiments API");

      experiments = data.map((exp: any) => ({
        id: exp.id,
        name: exp.name,
        description: exp.description || "",
        hyperparams: exp.hyperparams || [],
        tags: exp.tags || [],
        createdAt: new Date(exp.created_at),
        updatedAt: new Date(exp.updated_at),
        availableMetrics: exp.available_metrics || [],
        workspaceId: workspaceId,
      }));
    } catch (error) {
      console.error("Failed to load experiments:", error);
      errors.experiments =
        error instanceof Error ? error.message : "Failed to load experiments";
    } finally {
      loading.experiments = false;
    }
  }

  async function loadExperimentDetails(experimentId: string) {
    try {
      loading.experimentDetails = true;
      errors.experimentDetails = null;
      const response = await fetch(`/api/experiments/${experimentId}/metrics`);
      if (!response.ok)
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      const apiResponse = await response.json();
      const metrics = apiResponse.data;
      if (!metrics || !Array.isArray(metrics))
        throw new Error("Invalid response structure from metrics API");

      scalarMetrics = metrics;
    } catch (error) {
      console.error("Failed to load experiment details:", error);
      errors.experimentDetails =
        error instanceof Error
          ? error.message
          : "Failed to load experiment details";
    } finally {
      loading.experimentDetails = false;
    }
  }

  $effect(() => {
    loadWorkspaces();
  });

  $effect(() => {
    if (selectedWorkspace) {
      loadExperiments(selectedWorkspace.id);
      selectedExperiment = null;
      scalarMetrics = [];
    }
  });

  $effect(() => {
    if (selectedExperiment) {
      loadExperimentDetails(selectedExperiment.id);
    }
  });

  function onWorkspaceSelect(workspace: Workspace) {
    selectedWorkspace = workspace;
  }
  function onExperimentSelect(experiment: Experiment) {
    selectedExperiment = experiment;
  }
  function copyToClipboard(text: string) {
    navigator.clipboard.writeText(text);
  }
  function formatDate(date: Date): string {
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year:
        date.getFullYear() !== new Date().getFullYear() ? "numeric" : undefined,
    });
  }
</script>

<div class="bg-ctp-base text-ctp-text flex font-mono">
  <!-- =================================================================== -->
  <!-- Workspaces Panel (Column 1) - 25%                                   -->
  <!-- =================================================================== -->
  <div class="w-1/4 border-r border-ctp-surface0/30 flex flex-col">
    <!-- Header -->
    <div class="terminal-chrome-header">
      <div class="flex items-center justify-between mb-3">
        <h2 class="text-ctp-text font-medium text-base font-mono">
          workspaces
        </h2>
        <span
          class="bg-ctp-surface0/20 text-ctp-subtext0 px-2 py-1 text-xs font-mono border border-ctp-surface0/30"
          >[{workspaces.length}]</span
        >
      </div>
      <div
        class="flex items-center bg-ctp-surface0/20 focus-within:ring-1 focus-within:ring-ctp-text/20 transition-all"
      >
        <span class="text-ctp-subtext0 font-mono text-sm px-3 py-2">/</span>
        <input
          type="search"
          bind:value={workspaceSearchQuery}
          placeholder="search workspaces..."
          class="flex-1 bg-transparent border-0 py-2 pr-3 text-ctp-text placeholder-ctp-subtext0 focus:outline-none font-mono text-sm"
        />
      </div>
    </div>

    <!-- Scrollable Content -->
    <div class="flex-1 overflow-y-auto min-h-0">
      {#if loading.workspaces}
        <div class="text-center py-8 text-ctp-subtext0 text-sm font-mono">
          loading workspaces...
        </div>
      {:else if errors.workspaces}
        <div class="surface-layer-2 p-4 m-2">
          <div class="text-ctp-red font-medium mb-2 text-sm font-mono">
            error loading workspaces
          </div>
          <div class="text-ctp-subtext0 text-xs mb-3 font-mono">
            {errors.workspaces}
          </div>
          <button
            class="text-ctp-blue hover:text-ctp-blue/80 text-xs font-mono"
            onclick={() => loadWorkspaces()}>[retry]</button
          >
        </div>
      {:else if workspaces.length === 0}
        <div class="text-center py-8 text-ctp-subtext0 text-sm font-mono">
          no workspaces found
        </div>
      {:else}
        <div class="space-y-0">
          {#each workspaces.filter((w) => w.name
              .toLowerCase()
              .includes(workspaceSearchQuery.toLowerCase())) as workspace, index (workspace.id)}
            <button
              class="w-full text-left transition-all font-mono text-sm {selectedWorkspace?.id ===
              workspace.id
                ? 'bg-ctp-surface0/20 text-ctp-mauve border-l-2 border-l-ctp-mauve'
                : 'hover:bg-ctp-surface0/10 border-l-2 border-l-transparent hover:border-l-ctp-blue/30'} {index %
                2 ===
              0
                ? 'bg-ctp-surface0/5'
                : ''}"
              onclick={() => onWorkspaceSelect(workspace)}
            >
              <div class="p-3">
                <div class="flex items-center justify-between mb-2">
                  <span
                    class="font-medium text-ctp-text group-hover:text-ctp-blue transition-colors"
                    >{workspace.name}</span
                  >
                  <span class="text-xs text-ctp-subtext0 font-mono">
                    {workspace.role?.toLowerCase()}
                  </span>
                </div>
                {#if workspace.description}
                  <div class="text-xs text-ctp-subtext0 line-clamp-2 mb-2">
                    {workspace.description}
                  </div>
                {/if}
                <div class="text-xs text-ctp-overlay0">
                  {formatDate(workspace.createdAt)}
                </div>
              </div>
            </button>
          {/each}
        </div>
      {/if}
    </div>
  </div>

  <!-- =================================================================== -->
  <!-- Experiments Panel (Column 2) - 25%                                  -->
  <!-- =================================================================== -->
  <div class="w-1/4 border-r border-ctp-surface0/30 flex flex-col">
    <!-- Header -->
    <div class="terminal-chrome-header">
      {#if selectedWorkspace}
        <div class="flex items-center justify-between mb-3">
          <h2 class="text-ctp-text font-medium text-base font-mono">
            experiments
          </h2>
          <span
            class="bg-ctp-surface0/20 text-ctp-subtext0 px-2 py-1 text-xs font-mono border border-ctp-surface0/30"
            >[{experiments.length}]</span
          >
        </div>
        <div
          class="flex items-center bg-ctp-surface0/20 focus-within:ring-1 focus-within:ring-ctp-text/20 transition-all"
        >
          <span class="text-ctp-subtext0 font-mono text-sm px-3 py-2">/</span>
          <input
            type="search"
            bind:value={experimentSearchQuery}
            placeholder="search experiments..."
            class="flex-1 bg-transparent border-0 py-2 pr-3 text-ctp-text placeholder-ctp-subtext0 focus:outline-none font-mono text-sm"
          />
        </div>
      {:else}
        <div class="text-ctp-subtext0 text-sm font-mono">
          select a workspace to view experiments
        </div>
      {/if}
    </div>

    <!-- Scrollable Content -->
    <div class="flex-1 overflow-y-auto min-h-0">
      {#if selectedWorkspace}
        {#if loading.experiments}
          <div class="text-center py-8 text-ctp-subtext0 text-sm font-mono">
            loading experiments...
          </div>
        {:else if errors.experiments}
          <div class="surface-layer-2 p-4 m-2">
            <div class="text-ctp-red font-medium mb-2 text-sm font-mono">
              error loading experiments
            </div>
            <div class="text-ctp-subtext0 text-xs mb-3 font-mono">
              {errors.experiments}
            </div>
            <button
              class="text-ctp-blue hover:text-ctp-blue/80 text-xs font-mono"
              onclick={() =>
                selectedWorkspace && loadExperiments(selectedWorkspace.id)}
              >[retry]</button
            >
          </div>
        {:else if experiments.length === 0}
          <div class="text-center py-8 text-ctp-subtext0 text-sm font-mono">
            no experiments found
          </div>
        {:else}
          <div class="space-y-0">
            {#each experiments.filter((e) => e.name
                .toLowerCase()
                .includes(experimentSearchQuery.toLowerCase())) as experiment, index (experiment.id)}
              <button
                class="w-full text-left transition-all font-mono text-sm {selectedExperiment?.id ===
                experiment.id
                  ? 'bg-ctp-surface0/20 text-ctp-mauve border-l-2 border-l-ctp-mauve'
                  : 'hover:bg-ctp-surface0/10 border-l-2 border-l-transparent hover:border-l-ctp-blue/30'} {index %
                  2 ===
                0
                  ? 'bg-ctp-surface0/5'
                  : ''}"
                onclick={() => onExperimentSelect(experiment)}
              >
                <div class="p-3">
                  <div class="flex items-center justify-between mb-2">
                    <span
                      class="font-medium text-ctp-text group-hover:text-ctp-blue transition-colors"
                      >{experiment.name}</span
                    >
                    <span class="text-xs text-ctp-lavender"
                      >{formatDate(experiment.createdAt)}</span
                    >
                  </div>
                  {#if experiment.description}
                    <div class="text-xs text-ctp-subtext0 line-clamp-2 mb-2">
                      {experiment.description}
                    </div>
                  {/if}
                  <div
                    class="flex items-center space-x-3 text-xs text-ctp-overlay0"
                  >
                    {#if experiment.tags?.length}<span
                        >{experiment.tags.length} tags</span
                      >{/if}
                    {#if experiment.availableMetrics?.length}<span
                        >{experiment.availableMetrics.length} metrics</span
                      >{/if}
                  </div>
                </div>
              </button>
            {/each}
          </div>
        {/if}
      {/if}
    </div>
  </div>

  <!-- =================================================================== -->
  <!-- Details Panel (Column 3) - 50%                                      -->
  <!-- =================================================================== -->
  <div class="w-1/2 flex flex-col">
    <!-- Header -->
    <div class="terminal-chrome-header">
      {#if selectedExperiment}
        <div class="mb-3">
          <h2 class="text-ctp-text font-medium text-lg font-mono mb-2">
            {selectedExperiment.name}
          </h2>
          {#if selectedExperiment.description}
            <p class="text-ctp-subtext0 line-clamp-2 mb-3 text-sm">
              {selectedExperiment.description}
            </p>
          {/if}
          <button
            class="text-xs text-ctp-overlay0 hover:text-ctp-blue transition-colors font-mono"
            onclick={() =>
              selectedExperiment && copyToClipboard(selectedExperiment.id)}
            title="click to copy experiment id"
          >
            id: {selectedExperiment.id}
          </button>
        </div>
      {:else}
        <div class="text-ctp-subtext0 text-sm font-mono">
          select an experiment to view details
        </div>
      {/if}
    </div>

    <!-- Scrollable Content -->
    <div class="flex-1 overflow-y-auto p-4 min-h-0">
      {#if selectedExperiment}
        {#if loading.experimentDetails}
          <div class="text-center py-12">
            <div class="text-ctp-subtext0 text-sm font-mono">
              loading experiment details...
            </div>
          </div>
        {:else if errors.experimentDetails}
          <div class="surface-layer-2 p-4">
            <div class="text-ctp-red font-medium text-sm mb-3 font-mono">
              error loading experiment details
            </div>
            <div class="text-ctp-subtext0 mb-4 text-xs font-mono">
              {errors.experimentDetails}
            </div>
            <button
              class="text-ctp-blue hover:text-ctp-blue/80 transition-colors text-xs font-mono"
              onclick={() =>
                selectedExperiment &&
                loadExperimentDetails(selectedExperiment.id)}>[retry]</button
            >
          </div>
        {:else}
          <div class="space-y-6 font-mono">
            <!-- Metrics Section -->
            {#if scalarMetrics.length > 0}
              <div class="space-y-2">
                <div class="flex items-center gap-2">
                  <div class="text-sm text-ctp-text">metrics</div>
                  <div class="text-sm text-ctp-subtext0 font-mono">
                    [{scalarMetrics.length}]
                  </div>
                </div>
                <div class="terminal-chrome">
                  {#each scalarMetrics as metric, index}
                    <div
                      class="flex text-sm hover:bg-ctp-surface0/20 p-3 transition-colors {index !==
                      scalarMetrics.length - 1
                        ? 'border-b border-ctp-surface0/20'
                        : ''} {index % 2 === 0 ? 'bg-ctp-surface0/5' : ''}"
                    >
                      <div class="w-4 text-ctp-green">•</div>
                      <div
                        class="flex-1 text-ctp-text truncate"
                        title={metric.name}
                      >
                        {metric.name}
                      </div>
                      <div
                        class="w-24 text-right text-ctp-blue font-mono"
                        title={String(metric.value)}
                      >
                        {typeof metric.value === "number"
                          ? metric.value.toFixed(4)
                          : metric.value}
                      </div>
                    </div>
                  {/each}
                </div>
              </div>
            {/if}

            <!-- Tags Section -->
            {#if selectedExperiment.tags?.length}
              <div class="space-y-2">
                <div class="flex items-center gap-2">
                  <div class="text-sm text-ctp-text">tags</div>
                  <div class="text-sm text-ctp-subtext0 font-mono">
                    [{selectedExperiment.tags.length}]
                  </div>
                </div>
                <div class="flex flex-wrap gap-1">
                  {#each selectedExperiment.tags as tag}
                    <span
                      class="text-xs bg-ctp-blue/20 text-ctp-blue border border-ctp-blue/30 px-2 py-1 font-mono"
                      >{tag}</span
                    >
                  {/each}
                </div>
              </div>
            {/if}

            <!-- Hyperparameters Section -->
            {#if selectedExperiment.hyperparams?.length}
              <div class="space-y-2">
                <div class="flex items-center gap-2">
                  <div class="text-sm text-ctp-text">hyperparameters</div>
                  <div class="text-sm text-ctp-subtext0 font-mono">
                    [{selectedExperiment.hyperparams.length}]
                  </div>
                </div>
                <div class="terminal-chrome">
                  <div class="grid grid-cols-1 lg:grid-cols-2 gap-0">
                    {#each selectedExperiment.hyperparams as param, index}
                      <div
                        class="flex flex-col sm:flex-row sm:items-center sm:justify-between hover:bg-ctp-surface0/20 px-3 py-2 transition-colors text-sm gap-1 sm:gap-2 {index !==
                        selectedExperiment.hyperparams.length - 1
                          ? 'border-b border-ctp-surface0/20'
                          : ''} {index % 2 === 0 ? 'bg-ctp-surface0/5' : ''}"
                      >
                        <span class="text-ctp-subtext0 font-mono truncate"
                          >{param.key}</span
                        >
                        <span
                          class="text-ctp-blue font-mono bg-ctp-surface0/20 border border-ctp-surface0/30 px-2 py-1 max-w-32 truncate text-xs"
                          title={String(param.value)}>{param.value}</span
                        >
                      </div>
                    {/each}
                  </div>
                </div>
              </div>
            {/if}
          </div>
        {/if}
      {:else}
        <div class="flex items-center justify-center h-full">
          <div class="text-center text-ctp-subtext0 text-sm font-mono">
            select an experiment to view details
          </div>
        </div>
      {/if}
    </div>
  </div>
</div>
