<script lang="ts">
  import LandingPage from "$lib/components/landing-page.svelte";
  import DashboardCard from "$lib/components/dashboard-card.svelte";
  import RecentActivity from "$lib/components/recent-activity.svelte";
  import type { PageData } from "./$types";

  let { data }: { data: PageData } = $props();
  let { user, workspaces = [], recentExperiments = [], stats } = $derived(data);
</script>

{#if !user}
  <LandingPage />
{:else}
  <div class="text-ctp-text p-6 space-y-6">
    <div class="mb-8">
      <h1 class="text-2xl font-bold text-ctp-text mb-2">Dashboard</h1>
      <p class="text-ctp-subtext0">Welcome back! Here's what's happening with your experiments.</p>
    </div>

    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      <DashboardCard
        title="Total Experiments"
        value={stats?.totalExperiments || 0}
        description="Across all workspaces"
      />
      <DashboardCard
        title="Workspaces"
        value={stats?.totalWorkspaces || 0}
        description="Active workspaces"
        href="/workspaces"
      />
      <DashboardCard
        title="This Week"
        value={stats?.recentExperimentsCount || 0}
        description="New experiments"
      />
    </div>

    <div class="max-w-4xl mx-auto">
      <RecentActivity experiments={recentExperiments} workspaces={workspaces} />
    </div>
  </div>
{/if}
