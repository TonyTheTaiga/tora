import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import ExperimentDetailed from './experiment-detailed.svelte';
import { vi } from 'vitest';
import { writable } from 'svelte/store';
import '@testing-library/jest-dom/vitest';

// --- Mocks ---
vi.mock('$app/state', () => ({
  page: writable({ data: { user: { id: 'user123' } } }),
}));

const mockOpenDeleteExperimentModal = vi.fn();
const mockOpenEditExperimentModal = vi.fn();
const mockSetSelectedExperiment = vi.fn();

vi.mock('$lib/state/app.svelte.js', () => ({
  openDeleteExperimentModal: mockOpenDeleteExperimentModal,
  openEditExperimentModal: mockOpenEditExperimentModal,
  setSelectedExperiment: mockSetSelectedExperiment,
}));

// Mock global fetch
global.fetch = vi.fn();

// Mock lucide-svelte icons (similar to ExperimentSimple.test.ts)
vi.mock('lucide-svelte', async () => {
  const actual = await vi.importActual('lucide-svelte');
  const icons = {};
  for (const key in actual) {
    // Ensure we are trying to mock actual Svelte components
    if (actual[key] && (typeof actual[key] === 'function' || (typeof actual[key] === 'object' && typeof actual[key].render === 'function'))) {
      // Create a mock Svelte component
      const MockComponent = class {
        constructor(options) {
          // Store props if needed, e.g., for checking attributes like 'size'
          this.$$prop_def = options.props;
        }
        $destroy() {}
        $on() {}
        $set() {}
        // You might need $capture_state, $inject_state if your tests rely on component state
      };
      icons[key] = { default: MockComponent };
    } else {
      // For non-component exports, retain the actual value
      icons[key] = actual[key];
    }
  }
  return icons;
});


// Mock InteractiveChart component as it's complex and not the focus here
vi.mock('./interactive-chart.svelte', () => {
  // Create a basic Svelte component mock
  const MockInteractiveChart = class {
    constructor(options) { this.$$prop_def = options.props; }
    $destroy() {}
    $on() {}
    $set() {}
  };
  return { default: MockInteractiveChart };
});


describe('ExperimentDetailed.svelte', () => {
  const mockExperimentBase = {
    id: 'detail1',
    name: 'Detailed Experiment',
    description: 'A very detailed description.',
    createdAt: new Date().toISOString(),
    tags: ['detailTag1', 'detailTag2'],
    visibility: 'PUBLIC' as 'PUBLIC' | 'PRIVATE',
    user_id: 'user123',
    hyperparams: [{ key: 'param1', value: 'value1' }, { key: 'param2', value: 100 }],
    availableMetrics: ['metricA', 'metricB'],
    metricData: {
        'metricA': [1,2,3],
        'metricB': [4,5,6],
    },
  };

  beforeEach(() => {
    vi.clearAllMocks(); // Clears mock call counts etc.
    global.fetch.mockImplementation((url) => {
      const urlString = url.toString();
      if (urlString.includes('/api/experiments/') && urlString.includes('/ref')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(['ref1', 'ref2']) });
      }
      if (urlString.includes('/api/ai/analysis')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ hyperparameter_recommendations: {} }) });
      }
      if (urlString.includes('/api/experiments/') && urlString.includes('/metrics')) {
         return Promise.resolve({ ok: true, json: () => Promise.resolve([ { id: 'm1', name: 'metricA', value: 0.95, step: 100, created_at: new Date().toISOString() } ]) });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
    });
  });

  test('renders basic experiment information (name, ID, description, timestamp)', () => {
    render(ExperimentDetailed, { experiment: mockExperimentBase, highlighted: [] });

    expect(screen.getByText(mockExperimentBase.name)).toBeInTheDocument();
    expect(screen.getByText(mockExperimentBase.description)).toBeInTheDocument();
    expect(screen.getByText(`${mockExperimentBase.id.substring(0, 8)}...`)).toBeInTheDocument();
    const year = new Date(mockExperimentBase.createdAt).getFullYear().toString();
    expect(screen.getByText(expect.stringContaining(year))).toBeInTheDocument();
  });

  test('renders tags if present', () => {
    render(ExperimentDetailed, { experiment: mockExperimentBase, highlighted: [] });
    expect(screen.getByText('detailTag1')).toBeInTheDocument();
    expect(screen.getByText('detailTag2')).toBeInTheDocument();
  });

  test('renders hyperparameters if present and section is collapsible', async () => {
    render(ExperimentDetailed, { experiment: mockExperimentBase, highlighted: [] });
    const hyperparametersSection = screen.getByText('Hyperparameters').closest('summary');
    expect(hyperparametersSection).toBeInTheDocument();
    expect(hyperparametersSection?.parentElement).toHaveAttribute('open');
    expect(screen.getByText('param1')).toBeInTheDocument();
    expect(screen.getByText('value1')).toBeInTheDocument();

    if (hyperparametersSection) {
        await fireEvent.click(hyperparametersSection);
        expect(hyperparametersSection.parentElement).not.toHaveAttribute('open');
    }
  });

  test('renders metrics section if metrics are available and section is collapsible', async () => {
    render(ExperimentDetailed, { experiment: mockExperimentBase, highlighted: [] });
    const metricsSection = screen.getByText('Metrics').closest('summary');
    expect(metricsSection).toBeInTheDocument();
    expect(metricsSection?.parentElement).toHaveAttribute('open');
    expect(screen.getByText('Show Raw Data Table')).toBeInTheDocument();

     if (metricsSection) {
        await fireEvent.click(metricsSection);
        expect(metricsSection.parentElement).not.toHaveAttribute('open');
    }
  });

  test('metrics section: toggles between chart and raw data table', async () => {
    render(ExperimentDetailed, { experiment: mockExperimentBase, highlighted: [] });
    const toggleButton = screen.getByText('Show Raw Data Table');
    expect(screen.queryByText('Name')).not.toBeInTheDocument();

    await fireEvent.click(toggleButton);
    expect(await screen.findByText('Loading metrics...')).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText('Loading metrics...')).not.toBeInTheDocument());
    expect(await screen.findByText('Name')).toBeInTheDocument(); // Table header
    expect(toggleButton.textContent).toContain('Show Chart');

    await fireEvent.click(toggleButton);
    expect(screen.queryByText('Name')).not.toBeInTheDocument();
    expect(toggleButton.textContent).toContain('Show Raw Data Table');
  });

  test('metrics section: shows error if fetching metrics fails', async () => {
    global.fetch.mockImplementation((url) => {
      if (url.toString().includes(`/api/experiments/${mockExperimentBase.id}/metrics`)) {
        return Promise.resolve({ ok: false, statusText: 'Server Error Major Fail' });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
    });

    render(ExperimentDetailed, { experiment: { ...mockExperimentBase, metricData: {} }, highlighted: [] });
    const toggleButton = screen.getByText('Show Raw Data Table');
    await fireEvent.click(toggleButton);

    expect(await screen.findByText('Loading metrics...')).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText('Loading metrics...')).not.toBeInTheDocument());
    expect(await screen.findByText(/Failed to fetch metrics: Server Error Major Fail/i)).toBeInTheDocument();
  });

  test('action buttons call respective handlers or trigger actions', async () => {
    render(ExperimentDetailed, { experiment: mockExperimentBase, highlighted: [] });

    // AI Recommendations button
    const recommendButton = screen.getByTitle('Get AI recommendations');
    await fireEvent.click(recommendButton);
    expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining(`/api/ai/analysis?experimentId=${mockExperimentBase.id}`));

    // Edit button
    const editButton = screen.getByTitle('Edit experiment');
    await fireEvent.click(editButton);
    expect(mockOpenEditExperimentModal).toHaveBeenCalledWith(mockExperimentBase);

    // Show experiment chain button
    const chainButton = screen.getByTitle('Show experiment chain');
    await fireEvent.click(chainButton);
    expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining(`/api/experiments/${mockExperimentBase.id}/ref`));
    // Further state check for 'highlighted' would require more complex prop/state management in test

    // Delete button
    const deleteButton = screen.getByTitle('Delete experiment');
    await fireEvent.click(deleteButton);
    expect(mockOpenDeleteExperimentModal).toHaveBeenCalledWith(mockExperimentBase);

    // Minimize button
    const minimizeButton = screen.getByTitle('Minimize');
    await fireEvent.click(minimizeButton);
    expect(mockSetSelectedExperiment).toHaveBeenCalledWith(null);
  });

  test('action buttons (edit, delete, recommendations) are not shown for other users experiment', () => {
    const otherUsersExperiment = { ...mockExperimentBase, user_id: 'user457' };
    // page store mock already defines user as 'user123'
    render(ExperimentDetailed, { experiment: otherUsersExperiment, highlighted: [] });

    expect(screen.queryByTitle('Get AI recommendations')).not.toBeInTheDocument();
    expect(screen.queryByTitle('Edit experiment')).not.toBeInTheDocument();
    expect(screen.queryByTitle('Delete experiment')).not.toBeInTheDocument();

    // Chain and Minimize should still be there
    expect(screen.getByTitle('Show experiment chain')).toBeInTheDocument();
    expect(screen.getByTitle('Minimize')).toBeInTheDocument();
  });

  test('Copy ID button copies ID to clipboard', async () => {
    Object.assign(navigator, { clipboard: { writeText: vi.fn().mockResolvedValue(undefined) } });
    render(ExperimentDetailed, { experiment: mockExperimentBase, highlighted: [] });

    const copyIdButton = screen.getByLabelText('Copy Experiment ID');
    await fireEvent.click(copyIdButton);

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(mockExperimentBase.id);
    expect(await screen.findByText('Copied!')).toBeInTheDocument();
  });
});
