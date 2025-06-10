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

  const mockExperimentDetailedLong = {
    ...mockExperimentBase, // Spread the base mock
    id: 'detailLong1',
    name: 'Detailed Experiment With An Exceptionally Long Name That Goes On And On To Test How The UI Handles It And Whether Wrapping Or Truncation Occurs As Expected',
    description: 'This is an exceedingly long description for the detailed experiment view, designed specifically to test the multi-line truncation capabilities. It contains multiple sentences and should span several lines before being clamped by the CSS rules. We need to ensure that the title attribute is also correctly populated with this full verbose description for accessibility and user experience purposes, allowing users to see the complete text on hover when the visual representation is truncated by the line-clamping mechanism.',
    tags: ['AVeryLongTagNameForDetailedViewToTestTruncation', 'ShortDetailTag', 'AnotherQuiteLongDetailedTagThatWillAlsoBeTruncatedToEnsureConsistency'],
    hyperparams: [
        { key: 'a_very_long_hyperparameter_key_that_should_be_truncated', value: 'some_short_value' },
        { key: 'short_key', value: 'a_very_long_hyperparameter_value_that_is_a_string_and_should_also_be_truncated_to_prevent_layout_issues' },
        { key: 'numeric_param', value: 12345.6789 }
    ],
    // MetricData will be mocked via fetch for specific test
  };

  const mockRawMetricsWithLongString = [
    { id: 'm1', name: 'short_metric_name', value: 0.95, step: 100, created_at: new Date().toISOString() },
    { id: 'm2', name: 'metric_with_long_string_value', value: 'ThisIsAnExtremelyLongStringValueForAMetricThatShouldBeTruncatedInTheTableDisplayOtherwiseItWillBreakTheColumnWidthAndMakeThingsLookVeryBadIndeed', step: 1, created_at: new Date().toISOString() },
    { id: 'm3', name: 'metric_with_long_step_string', value: 0.50, step: 'Step_Alpha_Bravo_Charlie_Delta_Echo_Foxtrot_Golf_Hotel_India_Juliett_Kilo_Lima_Mike_November_Oscar_Papa_Quebec_Romeo_Sierra_Tango_Uniform_Victor_Whiskey_XRay_Yankee_Zulu', created_at: new Date().toISOString() },
  ];

  beforeEach(() => {
    vi.clearAllMocks(); // Clears mock call counts etc.
    // Reset global fetch to a more generic version or specific per test suite
    global.fetch = vi.fn((url) => {
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

  // --- Tests for new styling and truncation from previous step ---

  test('description has truncation class and title attribute for long descriptions', () => {
    const { container } = render(ExperimentDetailed, { experiment: mockExperimentDetailedLong, highlighted: [] });
    const descriptionElement = container.querySelector('p.description-truncate-detailed');
    expect(descriptionElement).toBeInTheDocument();
    expect(descriptionElement).toHaveClass('description-truncate-detailed');
    expect(descriptionElement).toHaveAttribute('title', mockExperimentDetailedLong.description);
  });

  test('tags have truncation classes and title attribute for long tag names', () => {
    const { container } = render(ExperimentDetailed, { experiment: mockExperimentDetailedLong, highlighted: [] });
    // Tags are in a div with class "flex gap-1.5 flex-wrap"
    // Each tag is a span with class "truncate max-w-[150px]"
    const tagElements = container.querySelectorAll('div.flex-wrap span.truncate');

    expect(tagElements.length).toBe(mockExperimentDetailedLong.tags.length);

    tagElements.forEach((tagElement, index) => {
      expect(tagElement).toHaveClass('truncate');
      expect(tagElement).toHaveClass('max-w-[150px]');
      expect(tagElement).toHaveAttribute('title', mockExperimentDetailedLong.tags[index]);
      expect(tagElement.textContent).toBe(mockExperimentDetailedLong.tags[index]);
    });
  });

  test('hyperparameter keys and string values are truncated and have title attributes', () => {
    const { container } = render(ExperimentDetailed, { experiment: mockExperimentDetailedLong, highlighted: [] });

    // Check first hyperparameter (long key)
    const longKeyElement = screen.getByText(mockExperimentDetailedLong.hyperparams[0].key);
    expect(longKeyElement).toHaveClass('truncate');
    expect(longKeyElement).toHaveAttribute('title', mockExperimentDetailedLong.hyperparams[0].key);

    const shortValueForLongKeyElement = screen.getByText(String(mockExperimentDetailedLong.hyperparams[0].value));
    // This one is short, but it also has truncate class by default
    expect(shortValueForLongKeyElement).toHaveClass('truncate');
    expect(shortValueForLongKeyElement).toHaveAttribute('title', String(mockExperimentDetailedLong.hyperparams[0].value));

    // Check second hyperparameter (long string value)
    const shortKeyElement = screen.getByText(mockExperimentDetailedLong.hyperparams[1].key);
    expect(shortKeyElement).toHaveClass('truncate');
    expect(shortKeyElement).toHaveAttribute('title', mockExperimentDetailedLong.hyperparams[1].key);

    const longValueElement = screen.getByText(String(mockExperimentDetailedLong.hyperparams[1].value));
    expect(longValueElement).toHaveClass('truncate');
    expect(longValueElement).toHaveAttribute('title', String(mockExperimentDetailedLong.hyperparams[1].value));
  });

  test('metrics table cells (value and step) are truncated for long string content', async () => {
    // Mock fetch for metrics data to return our long string metrics
    global.fetch.mockImplementation((url) => {
      if (url.toString().includes(`/api/experiments/${mockExperimentDetailedLong.id}/metrics`)) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockRawMetricsWithLongString),
        });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
    });

    render(ExperimentDetailed, { experiment: mockExperimentDetailedLong, highlighted: [] });

    // Open the metrics table
    const toggleButton = screen.getByText('Show Raw Data Table');
    await fireEvent.click(toggleButton);
    await waitFor(() => expect(screen.queryByText('Loading metrics...')).not.toBeInTheDocument());

    // Metric with long string value
    const longStringValueCell = screen.getByTitle(mockRawMetricsWithLongString[1].value as string);
    expect(longStringValueCell).toBeInTheDocument();
    expect(longStringValueCell).toHaveClass('truncate');
    expect(longStringValueCell).toHaveClass('max-w-sm');
    expect(longStringValueCell.textContent).toBe(mockRawMetricsWithLongString[1].value);


    // Metric with long step string
    // The cell content will be the long step string.
    const longStepCell = screen.getAllByText(mockRawMetricsWithLongString[2].step as string)[0]; // Get by full text content
    expect(longStepCell).toBeInTheDocument();
    expect(longStepCell).toHaveClass('truncate');
    expect(longStepCell).toHaveClass('max-w-[70px]');
    // Note: The title attribute for step is not explicitly set in the component code,
    // so we don't check for it here. The browser might add one if text is visually truncated.
  });
});
