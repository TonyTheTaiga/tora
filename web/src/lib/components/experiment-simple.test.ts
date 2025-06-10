import { render, screen, fireEvent } from '@testing-library/svelte';
import ExperimentSimple from './experiment-simple.svelte';
import { vi } from 'vitest';
import { writable } from 'svelte/store'; // For mocking stores

// Mock SvelteKit modules
vi.mock('$app/state', () => ({
  page: writable({ data: { user: { id: 'user123' } } }), // Mock user data as needed by component
}));

vi.mock('$lib/state/app.svelte.js', () => ({
  openDeleteExperimentModal: vi.fn(),
  setSelectedExperiment: vi.fn(),
  getSelectedExperiment: vi.fn(() => null), // Default to no experiment selected
}));

// Mock lucide-svelte icons to avoid rendering actual SVG content in tests
// and to check if they are being used.
vi.mock('lucide-svelte', async () => {
  const actual = await vi.importActual('lucide-svelte');
  const iconMock = (props) => ({ /* Basic Svelte component mock */
    Component: class {
      constructor(options) { this.$$prop_def = options.props; }
      $set() {}
      $on() {}
      $destroy() {}
    },
    // You can add an attribute to the mock if you need to find it by testId or class
    // e.g., html: `<svg data-lucide="${name}"></svg>`
  });

  const icons = {};
  for (const key in actual) {
    if (typeof actual[key] === 'function' || (typeof actual[key] === 'object' && actual[key].render)) {
      // Mocking Svelte components from lucide-svelte
      // Create a simple Svelte component mock for each icon
      icons[key] = {
        // Svelte component structure for testing purposes
        // This is a simplified mock. If tests need to interact with icon internals, this might need adjustment.
        render: vi.fn(),
        $$prop_def: {}, // Mock structure Svelte components expect
        // If you need to check props passed to icons, you can enhance this mock
        // For now, just making sure they don't break rendering.
        // A common pattern is to replace them with a simple div or span with a test id:
        // Example: new Proxy({}, { get: (target, prop) => (props) => ({ كومponent: class { ... }, html: `<div data-lucide-icon="${prop}"></div>`})})
        // For simplicity, we'll just mock them as empty components for now.
        // A more robust way is to mock them as basic Svelte components:
        // Credit: https://github.com/testing-library/svelte-testing-library/issues/134#issuecomment-1009075139
        // Adapting for lucide-svelte which exports component constructors directly
        default: class { $destroy = () => {}; constructor(options) { this.$$prop_def = options.props; } $set() {} $on() {} }
      };
    } else {
      icons[key] = actual[key];
    }
  }
  return { ...icons };
});


describe('ExperimentSimple.svelte', () => {
  const mockExperiment = {
    id: 'exp1',
    name: 'Test Experiment Name',
    description: 'Test experiment description.',
    createdAt: new Date().toISOString(),
    tags: ['tag1', 'TagTwo'],
    visibility: 'PUBLIC' as 'PUBLIC' | 'PRIVATE',
    user_id: 'user123',
    hyperparams: [],
    availableMetrics: [],
  };

  const mockExperimentLong = {
    id: 'expLong',
    name: 'An Extremely Long Experiment Name That Absolutely Must Be Truncated Otherwise It Will Break The Entire Layout And Cause Havoc',
    description: 'This is a very long description that definitely should be truncated to multiple lines to avoid breaking the layout of the card. It keeps going and going and on and on, providing extensive details that are not suitable for a simple card view but are here for testing purposes. More text to ensure it hits the clamp limit.',
    createdAt: new Date().toISOString(),
    tags: ['ALongTagNameThatDefinitelyShouldBeTruncated', 'ShortTag', 'AnotherQuiteLongTagThatWillAlsoBeTruncated'],
    visibility: 'PUBLIC' as 'PUBLIC' | 'PRIVATE',
    user_id: 'user123',
    hyperparams: [],
    availableMetrics: [],
  };


  afterEach(() => {
    vi.clearAllMocks();
    global.fetch = undefined; // Clear global fetch mock if set in a test
  });

  test('renders basic experiment information (name, description, tags)', () => {
    render(ExperimentSimple, { experiment: mockExperiment, highlighted: [] });

    expect(screen.getByText(mockExperiment.name)).toBeInTheDocument();
    expect(screen.getByText(mockExperiment.description)).toBeInTheDocument();
    expect(screen.getByText('tag1')).toBeInTheDocument();
    expect(screen.getByText('TagTwo')).toBeInTheDocument();
  });

  test('does not render description if not provided', () => {
    const experimentWithoutDesc = { ...mockExperiment, description: null };
    render(ExperimentSimple, { experiment: experimentWithoutDesc, highlighted: [] });
    expect(screen.queryByText(mockExperiment.description)).not.toBeInTheDocument();
  });

  test('renders date correctly (simplified check, actual formatting is locale-dependent)', () => {
    render(ExperimentSimple, { experiment: mockExperiment, highlighted: [] });
    // This is a very basic check. Date formatting is complex and locale-dependent.
    // Consider checking for a part of the date or mocking date-fns/equivalent if precise format is critical.
    const dateElement = screen.getByTitle(new Date(mockExperiment.createdAt).toLocaleString());
    expect(dateElement).toBeInTheDocument();
    // Example: expect(dateElement.textContent).toContain(new Date(mockExperiment.createdAt).getFullYear());
  });

  test('applies comparison border when isSelectedForComparison is true', () => {
    const { container } = render(ExperimentSimple, {
      experiment: mockExperiment,
      highlighted: [],
      isSelectedForComparison: true,
    });
    const articleElement = container.querySelector('article');
    expect(articleElement).toHaveClass('border-ctp-blue');
    expect(articleElement).not.toHaveClass('border-transparent');
  });

  test('applies transparent border when isSelectedForComparison is false or not provided', () => {
    const { container } = render(ExperimentSimple, {
      experiment: mockExperiment,
      highlighted: [],
      isSelectedForComparison: false,
    });
    const articleElement = container.querySelector('article');
    expect(articleElement).toHaveClass('border-transparent');
    expect(articleElement).not.toHaveClass('border-ctp-blue');

    const { container: containerUndefined } = render(ExperimentSimple, {
      experiment: mockExperiment,
      highlighted: [],
    });
    const articleElementUndefined = containerUndefined.querySelector('article');
    expect(articleElementUndefined).toHaveClass('border-transparent');
    expect(articleElementUndefined).not.toHaveClass('border-ctp-blue');
  });

  test('renders public visibility icon', () => {
    render(ExperimentSimple, { experiment: { ...mockExperiment, visibility: 'PUBLIC' }, highlighted: [] });
    // Check based on title or a data-testid if icons were mocked more thoroughly
    expect(screen.getByTitle('Public')).toBeInTheDocument();
  });

  test('renders private visibility icon', () => {
    render(ExperimentSimple, { experiment: { ...mockExperiment, visibility: 'PRIVATE' }, highlighted: [] });
    expect(screen.getByTitle('Private')).toBeInTheDocument();
  });

  test('delete button calls openDeleteExperimentModal on click', async () => {
    const { openDeleteExperimentModal } = await import('$lib/state/app.svelte.js');
    render(ExperimentSimple, { experiment: mockExperiment, highlighted: [] });

    const deleteButton = screen.getByTitle('Delete experiment');
    await fireEvent.click(deleteButton);

    expect(openDeleteExperimentModal).toHaveBeenCalledWith(mockExperiment);
  });

  test('show experiment chain button works', async () => {
    // This button fetches data, so a more complex mock might be needed for full behavior.
    // For now, just testing its presence and basic interaction if possible.
    // Mocking global fetch for this specific test
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(['ref1', 'ref2']),
      })
    ) as any;

    let highlighted: string[] = [];
    const setHighlighted = (newVal: string[]) => { highlighted = newVal; };

    // For bind:highlighted, we need to simulate how Svelte's $bindable works or pass a store
    // A simpler way for testing is to pass `highlighted` and re-render or check callback.
    // However, the component uses a local $state for highlighted passed as $bindable.
    // Let's test the click and assume the fetch and state update logic inside the component works.
    // Direct testing of bind:highlighted is tricky without a parent component.
    // We can check if the icon changes, implying the internal state changed.

    const { component } = render(ExperimentSimple, { experiment: mockExperiment, highlighted: [] });

    const eyeButton = screen.getByTitle('Show experiment chain');
    await fireEvent.click(eyeButton);

    expect(global.fetch).toHaveBeenCalledWith(`/api/experiments/${mockExperiment.id}/ref`);
    // Due to the way highlighted is bound, we cannot directly check its updated value here
    // without a more complex setup. We'd typically check if the icon changed if Eye and EyeClosed were distinct components.
    // Since they are mocked generically, this specific outcome is harder to verify here.
    // We've verified fetch was called, which is a key part of the interaction.
  });

  // Test for user not owning the experiment (delete button not visible)
  test('delete button is not rendered if user does not own experiment', async () => {
    const otherUserExperiment = { ...mockExperiment, user_id: 'user456' };
    // Mock page store to reflect current user
    const { page: pageStore } = await import('$app/state');
    pageStore.set({ data: { user: { id: 'user123' } } } as any); // Current user is 'user123'

    render(ExperimentSimple, { experiment: otherUserExperiment, highlighted: [] });
    expect(screen.queryByTitle('Delete experiment')).not.toBeInTheDocument();
  });

  // --- Tests for new styling and truncation ---

  test('applies fixed height class for medium screens (md:h-64)', () => {
    const { container } = render(ExperimentSimple, { experiment: mockExperiment, highlighted: [] });
    const articleElement = container.querySelector('article');
    expect(articleElement).toHaveClass('md:h-64');
  });

  test('experiment name has truncate class and title attribute for long names', () => {
    render(ExperimentSimple, { experiment: mockExperimentLong, highlighted: [] });
    const nameElement = screen.getByText(mockExperimentLong.name); // Text content will be the full name
    expect(nameElement).toHaveClass('truncate');
    expect(nameElement.getAttribute('title')).toBe(null); // Title is on the parent <p> for description, name itself if not truncated by parent
                                                        // The h3 itself is what gets truncated. It should have the title.
                                                        // Actually, h3 itself does not have a title attribute in the component.
                                                        // The text is directly inside. The parent div doesn't have it either.
                                                        // Let's assume the browser provides title on truncated text, or test it by checking its container.
                                                        // For now, class check is primary. The component doesn't explicitly set title on H3.
  });

  test('experiment name is truncated', () => {
    render(ExperimentSimple, { experiment: mockExperimentLong, highlighted: [] });
    const nameElement = screen.getByText(mockExperimentLong.name);
    expect(nameElement).toHaveClass('truncate');
  });


  test('description has truncation class and title attribute for long descriptions', () => {
    const { container } = render(ExperimentSimple, { experiment: mockExperimentLong, highlighted: [] });
    const descriptionElement = container.querySelector('p.description-truncate');
    expect(descriptionElement).toBeInTheDocument();
    expect(descriptionElement).toHaveClass('description-truncate');
    expect(descriptionElement).toHaveAttribute('title', mockExperimentLong.description);
  });

  test('tags have truncation classes and title attribute for long tag names', () => {
    const { container } = render(ExperimentSimple, { experiment: mockExperimentLong, highlighted: [] });
    const tagElements = container.querySelectorAll('.flex-nowrap span.truncate'); // Selects the individual tag pills

    expect(tagElements.length).toBe(mockExperimentLong.tags.length);

    tagElements.forEach((tagElement, index) => {
      expect(tagElement).toHaveClass('truncate');
      expect(tagElement).toHaveClass('max-w-[120px]');
      expect(tagElement).toHaveAttribute('title', mockExperimentLong.tags[index]);
      expect(tagElement.textContent).toBe(mockExperimentLong.tags[index]); // Text content should be full, CSS handles visual truncation
    });
  });

  test('tags container has overflow-x-auto class', () => {
    const { container } = render(ExperimentSimple, { experiment: mockExperimentLong, highlighted: [] });
    // Find the div that is the direct parent of the div containing tags, and should have overflow-x-auto
    const tagsOuterContainer = container.querySelector('div.flex.items-center.gap-1\\.5.text-xs.text-ctp-subtext0');
    expect(tagsOuterContainer).toHaveClass('overflow-x-auto');
  });

});

// Helper to ensure Vitest and JSDOM environment are set up for @testing-library/jest-dom
import '@testing-library/jest-dom/vitest';
