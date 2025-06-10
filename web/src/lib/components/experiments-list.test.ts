import { render, screen, fireEvent } from '@testing-library/svelte';
import ExperimentsList from './experiments-list.svelte';
import { vi } from 'vitest';
import { writable, get } from 'svelte/store'; // Import `get` for reading store value in mocks
import '@testing-library/jest-dom/vitest';

// --- Mocks ---

// Mock child components: ExperimentSimple and ExperimentDetailed
// We want to check if they are rendered and what props they receive,
// not their internal behavior (that's tested separately).
vi.mock('./experiment-simple.svelte', () => ({
  default: vi.fn().mockImplementation((options) => {
    // A simple mock that can render something identifiable or check props
    const { props } = options;
    // You could create a DOM element here if needed for assertions:
    // const el = document.createElement('div');
    // el.textContent = `Simple: ${props.experiment.name}`;
    // el.setAttribute('data-testid', 'experiment-simple');
    // if(props.isSelectedForComparison) el.setAttribute('data-selected-comparison', 'true');
    // options.target.appendChild(el); // This part is tricky with Svelte Testing Library's render
    // For STL, it's often better to let the library handle rendering and just mock the component's interface.
    return {
      // Mock Svelte component lifecycle methods and props
      $$prop_def: props, // Makes props available on the component instance
      $on: vi.fn(),
      $destroy: vi.fn(),
      $set: vi.fn(),
      // If the component uses bind:prop, you might need to mock that behavior if the parent relies on it.
    };
  }),
}));

vi.mock('./experiment-detailed.svelte', () => ({
  default: vi.fn().mockImplementation((options) => {
    const { props } = options;
    return {
      $$prop_def: props,
      $on: vi.fn(),
      $destroy: vi.fn(),
      $set: vi.fn(),
    };
  }),
}));


// Mock SvelteKit modules and app state
const mockGetMode = vi.fn(() => false); // Default to not in comparison mode
const mockAddExperiment = vi.fn();
const mockSelectedForComparison = vi.fn(() => false);

vi.mock('$lib/state/comparison.svelte.js', () => ({
  getMode: mockGetMode,
  addExperiment: mockAddExperiment,
  selectedForComparison: mockSelectedForComparison,
}));

const mockSelectedExperimentStore = writable(null);
const mockSetSelectedExperiment = vi.fn((exp) => mockSelectedExperimentStore.set(exp));

vi.mock('$lib/state/app.svelte.js', () => ({
  getSelectedExperiment: () => get(mockSelectedExperimentStore), // Return the current value of the mock store
  setSelectedExperiment: mockSetSelectedExperiment,
  // Other modals are not directly interacted with by ExperimentsList, so simple mocks suffice
  getCreateExperimentModal: writable(null),
  getEditExperimentModal: writable(null),
  getDeleteExperimentModal: writable(null),
}));


describe('ExperimentsList.svelte', () => {
  const mockExperimentsData = [
    { id: '1', name: 'Alpha Exp', description: 'First one', user_id: 'user1', createdAt: new Date().toISOString(), visibility: 'PUBLIC', tags: [], hyperparams: [], availableMetrics: [] },
    { id: '2', name: 'Beta Exp', description: 'Second one', user_id: 'user1', createdAt: new Date().toISOString(), visibility: 'PRIVATE', tags: [], hyperparams: [], availableMetrics: [] },
    { id: '3', name: 'Gamma Exp', description: 'Third one', user_id: 'user1', createdAt: new Date().toISOString(), visibility: 'PUBLIC', tags: [], hyperparams: [], availableMetrics: [] },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    mockSelectedExperimentStore.set(null); // Reset selected experiment before each test
    mockGetMode.mockReturnValue(false); // Default to not comparison mode
    mockSelectedForComparison.mockReturnValue(false); // Default to not selected for comparison
    // Reset mocks for child components if they were introspected (e.g., call counts)
    vi.mocked(await import('./experiment-simple.svelte')).default.mockClear();
    vi.mocked(await import('./experiment-detailed.svelte')).default.mockClear();
  });

  test('renders a list of ExperimentSimple components', async () => {
    render(ExperimentsList, { experiments: mockExperimentsData });
    const ExperimentSimpleMock = (await import('./experiment-simple.svelte')).default;
    expect(ExperimentSimpleMock).toHaveBeenCalledTimes(mockExperimentsData.length);

    mockExperimentsData.forEach(exp => {
      expect(ExperimentSimpleMock).toHaveBeenCalledWith(
        expect.objectContaining({
          props: expect.objectContaining({ experiment: exp })
        })
      );
    });
  });

  test('renders empty state or nothing if no experiments are provided (currently renders nothing)', () => {
    render(ExperimentsList, { experiments: [] });
    // The component currently renders an empty div if experiments array is empty.
    // We check that no ExperimentSimple components are rendered.
    expect(vi.mocked(require('./experiment-simple.svelte').default)).not.toHaveBeenCalled();
  });

  test('clicking an experiment calls setSelectedExperiment when not in comparison mode', async () => {
    mockGetMode.mockReturnValue(false);
    const { getAllByRole } = render(ExperimentsList, { experiments: mockExperimentsData });

    // All experiment items have role="button" on their wrapper div
    const experimentItems = getAllByRole('button');
    await fireEvent.click(experimentItems[0]);

    expect(mockSetSelectedExperiment).toHaveBeenCalledWith(mockExperimentsData[0]);
    expect(mockAddExperiment).not.toHaveBeenCalled();
  });

  test('clicking an experiment calls addExperiment when in comparison mode', async () => {
    mockGetMode.mockReturnValue(true); // Set to comparison mode
    const { getAllByRole } = render(ExperimentsList, { experiments: mockExperimentsData });

    const experimentItems = getAllByRole('button');
    await fireEvent.click(experimentItems[1]);

    expect(mockAddExperiment).toHaveBeenCalledWith(mockExperimentsData[1].id);
    expect(mockSetSelectedExperiment).not.toHaveBeenCalled();
  });

  test('renders ExperimentDetailed when an experiment is selected', async () => {
    render(ExperimentsList, { experiments: mockExperimentsData });
    const ExperimentSimpleMock = (await import('./experiment-simple.svelte')).default;
    const ExperimentDetailedMock = (await import('./experiment-detailed.svelte')).default;

    // Initial render: all simple
    expect(ExperimentSimpleMock).toHaveBeenCalledTimes(3);
    expect(ExperimentDetailedMock).not.toHaveBeenCalled();

    // Simulate selecting an experiment
    mockSelectedExperimentStore.set(mockExperimentsData[0]);

    // Wait for Svelte's reactivity to update the component
    await waitFor(() => {
      // Now one should be detailed, others simple
      expect(ExperimentSimpleMock).toHaveBeenCalledTimes(2);
      expect(ExperimentDetailedMock).toHaveBeenCalledTimes(1);
      expect(ExperimentDetailedMock).toHaveBeenCalledWith(
        expect.objectContaining({
          props: expect.objectContaining({ experiment: mockExperimentsData[0] })
        })
      );
    });
  });

  test('applies opacity-40 to non-highlighted experiments', async () => {
    const { container } = render(ExperimentsList, {
      experiments: mockExperimentsData,
      highlighted: [mockExperimentsData[0].id]
    });

    // The wrapper div for each experiment simple instance is what gets the opacity class
    const experimentWrappers = container.querySelectorAll('div[id^="experiment-"] > div:first-child');

    // Wrapper for exp1 (highlighted) should NOT have opacity-40
    // This assumes the structure div#experiment-X > div.cursor-pointer.group
    const wrapperExp1 = container.querySelector(`#experiment-${mockExperimentsData[0].id} > div:first-child`);
    expect(wrapperExp1).not.toHaveClass('opacity-40');

    // Wrapper for exp2 (not highlighted) SHOULD have opacity-40
    const wrapperExp2 = container.querySelector(`#experiment-${mockExperimentsData[1].id} > div:first-child`);
    expect(wrapperExp2).toHaveClass('opacity-40');

    // Wrapper for exp3 (not highlighted) SHOULD have opacity-40
    const wrapperExp3 = container.querySelector(`#experiment-${mockExperimentsData[2].id} > div:first-child`);
    expect(wrapperExp3).toHaveClass('opacity-40');
  });

  test('passes isSelectedForComparison prop correctly to ExperimentSimple', async () => {
    mockSelectedForComparison.mockImplementation(id => id === mockExperimentsData[1].id); // Exp2 is selected for comparison
    render(ExperimentsList, { experiments: mockExperimentsData });
    const ExperimentSimpleMock = (await import('./experiment-simple.svelte')).default;

    expect(ExperimentSimpleMock).toHaveBeenCalledWith(
      expect.objectContaining({
        props: expect.objectContaining({ experiment: mockExperimentsData[0], isSelectedForComparison: false })
      })
    );
    expect(ExperimentSimpleMock).toHaveBeenCalledWith(
      expect.objectContaining({
        props: expect.objectContaining({ experiment: mockExperimentsData[1], isSelectedForComparison: true })
      })
    );
    expect(ExperimentSimpleMock).toHaveBeenCalledWith(
      expect.objectContaining({
        props: expect.objectContaining({ experiment: mockExperimentsData[2], isSelectedForComparison: false })
      })
    );
  });
});
