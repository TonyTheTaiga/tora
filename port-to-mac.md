# Tora Native macOS Implementation Plan

## Application Overview
Tora is an ML experiment tracking and visualization platform. This document outlines the 1:1 functional implementation for recreating it as a native macOS Swift application.

## Core Data Models

### 1. Workspace
```swift
@Model
class Workspace {
    var id: String
    var name: String
    var description: String?
    var createdAt: Date
    var role: WorkspaceRole // VIEWER, EDITOR, ADMIN, OWNER
    var experiments: [Experiment] = []
    var members: [WorkspaceMember] = []
}

enum WorkspaceRole: String, CaseIterable {
    case viewer = "VIEWER"
    case editor = "EDITOR" 
    case admin = "ADMIN"
    case owner = "OWNER"
}
```

### 2. Experiment
```swift
@Model
class Experiment {
    var id: String
    var name: String
    var description: String
    var hyperparams: [HyperParam] = []
    var tags: [String] = []
    var createdAt: Date
    var updatedAt: Date
    var visibility: Visibility // PUBLIC, PRIVATE
    var status: ExperimentStatus // COMPLETED, RUNNING, FAILED
    var startedAt: Date?
    var endedAt: Date?
    var createdBy: String?
    var version: String?
    var metrics: [Metric] = []
    var workspace: Workspace?
    var availableMetrics: [String] = []
}

enum Visibility: String, CaseIterable {
    case public = "PUBLIC"
    case private = "PRIVATE"
}

enum ExperimentStatus: String, CaseIterable {
    case completed = "COMPLETED"
    case running = "RUNNING"
    case failed = "FAILED"
}
```

### 3. Metric
```swift
@Model
class Metric {
    var id: Int
    var experimentId: String
    var name: String
    var value: Double
    var step: Int?
    var metadata: [String: Any]?
    var createdAt: Date
    var experiment: Experiment?
}
```

### 4. HyperParam
```swift
@Model
class HyperParam {
    var key: String
    var value: String // Store as string, parse as needed
}
```

### 5. API Key
```swift
@Model
class APIKey {
    var id: String
    var name: String
    var keyHash: String
    var createdAt: Date
    var lastUsed: Date
    var revoked: Bool
}
```

## SwiftUI View Structure

### 1. App Entry Point
```swift
@main
struct ToraApp: App {
    @StateObject private var authManager = AuthenticationManager()
    @StateObject private var workspaceManager = WorkspaceManager()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(authManager)
                .environmentObject(workspaceManager)
        }
        .modelContainer(for: [Workspace.self, Experiment.self, Metric.self])
    }
}
```

### 2. Main Navigation
```swift
struct ContentView: View {
    @EnvironmentObject var authManager: AuthenticationManager
    
    var body: some View {
        Group {
            if authManager.isAuthenticated {
                MainTabView()
            } else {
                AuthenticationView()
            }
        }
    }
}

struct MainTabView: View {
    var body: some View {
        NavigationSplitView {
            WorkspaceListView()
        } detail: {
            WorkspaceDetailView()
        }
    }
}
```

### 3. Authentication Views
```swift
struct AuthenticationView: View {
    @State private var showingSignup = false
    
    var body: some View {
        if showingSignup {
            SignupView()
        } else {
            LoginView()
        }
    }
}

struct LoginView: View {
    @State private var email = ""
    @State private var password = ""
    @EnvironmentObject var authManager: AuthenticationManager
    
    var body: some View {
        VStack(spacing: 20) {
            TextField("Email", text: $email)
            SecureField("Password", text: $password)
            Button("Login") {
                authManager.login(email: email, password: password)
            }
        }
        .padding()
    }
}
```

### 4. Workspace Views
```swift
struct WorkspaceListView: View {
    @Query private var workspaces: [Workspace]
    @State private var showingCreateWorkspace = false
    
    var body: some View {
        List(workspaces) { workspace in
            NavigationLink(destination: WorkspaceDetailView(workspace: workspace)) {
                WorkspaceRowView(workspace: workspace)
            }
        }
        .navigationTitle("Workspaces")
        .toolbar {
            Button("Create Workspace") {
                showingCreateWorkspace = true
            }
        }
        .sheet(isPresented: $showingCreateWorkspace) {
            CreateWorkspaceView()
        }
    }
}

struct WorkspaceDetailView: View {
    let workspace: Workspace
    @State private var searchText = ""
    @State private var selectedExperiment: Experiment?
    
    var filteredExperiments: [Experiment] {
        if searchText.isEmpty {
            return workspace.experiments
        }
        return workspace.experiments.filter { experiment in
            experiment.name.localizedCaseInsensitiveContains(searchText) ||
            experiment.description.localizedCaseInsensitiveContains(searchText) ||
            experiment.tags.contains { $0.localizedCaseInsensitiveContains(searchText) }
        }
    }
    
    var body: some View {
        NavigationSplitView {
            ExperimentListView(experiments: filteredExperiments)
                .searchable(text: $searchText)
        } detail: {
            if let selectedExperiment = selectedExperiment {
                ExperimentDetailView(experiment: selectedExperiment)
            } else {
                Text("Select an experiment")
            }
        }
        .navigationTitle(workspace.name)
    }
}
```

### 5. Experiment Views
```swift
struct ExperimentListView: View {
    let experiments: [Experiment]
    @State private var showingCreateExperiment = false
    
    var body: some View {
        List(experiments) { experiment in
            NavigationLink(destination: ExperimentDetailView(experiment: experiment)) {
                ExperimentRowView(experiment: experiment)
            }
        }
        .toolbar {
            Button("Create Experiment") {
                showingCreateExperiment = true
            }
        }
        .sheet(isPresented: $showingCreateExperiment) {
            CreateExperimentView()
        }
    }
}

struct ExperimentDetailView: View {
    let experiment: Experiment
    @State private var selectedMetrics: [String] = []
    @State private var showingChart = true
    
    var scalarMetrics: [Metric] {
        // Group metrics by name, return those with single values
        let grouped = Dictionary(grouping: experiment.metrics, by: { $0.name })
        return grouped.compactMap { (name, metrics) in
            metrics.count == 1 ? metrics.first : nil
        }
    }
    
    var timeSeriesMetrics: [String: [Metric]] {
        // Group metrics by name, return those with multiple values
        let grouped = Dictionary(grouping: experiment.metrics, by: { $0.name })
        return grouped.filter { $0.value.count > 1 }
    }
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Experiment header
                ExperimentHeaderView(experiment: experiment)
                
                // Metrics toggle
                Picker("View", selection: $showingChart) {
                    Text("Chart").tag(true)
                    Text("Data").tag(false)
                }
                .pickerStyle(SegmentedPickerStyle())
                
                if showingChart {
                    // Interactive chart
                    InteractiveChartView(
                        experiment: experiment,
                        selectedMetrics: $selectedMetrics
                    )
                } else {
                    // Metrics data view
                    MetricsDataView(
                        scalarMetrics: scalarMetrics,
                        timeSeriesMetrics: timeSeriesMetrics
                    )
                }
                
                // Hyperparameters
                HyperParametersView(hyperparams: experiment.hyperparams)
                
                // System info
                SystemInfoView(experiment: experiment)
            }
            .padding()
        }
        .navigationTitle(experiment.name)
    }
}
```

### 6. Chart Views
```swift
struct InteractiveChartView: View {
    let experiment: Experiment
    @Binding var selectedMetrics: [String]
    @State private var searchFilter = ""
    
    var availableMetrics: [String] {
        experiment.availableMetrics.filter { metric in
            searchFilter.isEmpty || metric.localizedCaseInsensitiveContains(searchFilter)
        }
    }
    
    var body: some View {
        VStack {
            // Metric selector
            MetricSelectorView(
                availableMetrics: availableMetrics,
                selectedMetrics: $selectedMetrics,
                searchFilter: $searchFilter
            )
            
            // Chart
            if !selectedMetrics.isEmpty {
                TimeSeriesChartView(
                    experiment: experiment,
                    selectedMetrics: selectedMetrics
                )
                .frame(height: 400)
            } else {
                Text("Select metrics to visualize")
                    .foregroundColor(.secondary)
                    .frame(height: 400)
            }
        }
    }
}

struct TimeSeriesChartView: View {
    let experiment: Experiment
    let selectedMetrics: [String]
    
    var chartData: [(String, [Metric])] {
        let grouped = Dictionary(grouping: experiment.metrics, by: { $0.name })
        return selectedMetrics.compactMap { metricName in
            guard let metrics = grouped[metricName] else { return nil }
            return (metricName, metrics.sorted { $0.step ?? 0 < $1.step ?? 0 })
        }
    }
    
    var body: some View {
        Chart {
            ForEach(chartData, id: \.0) { (metricName, metrics) in
                ForEach(Array(metrics.enumerated()), id: \.offset) { index, metric in
                    LineMark(
                        x: .value("Step", metric.step ?? index),
                        y: .value("Value", metric.value)
                    )
                    .foregroundStyle(by: .value("Metric", metricName))
                }
            }
        }
        .chartXAxisLabel("Step")
        .chartYAxisLabel("Value")
    }
}
```

### 7. Comparison Views
```swift
struct ComparisonView: View {
    @State private var selectedExperiments: [Experiment] = []
    @State private var selectedMetrics: [String] = []
    
    var commonMetrics: [String] {
        guard !selectedExperiments.isEmpty else { return [] }
        
        let metricSets = selectedExperiments.map { experiment in
            Set(experiment.metrics.map { $0.name })
        }
        
        return Array(metricSets.reduce(metricSets.first ?? Set()) { result, set in
            result.intersection(set)
        }).sorted()
    }
    
    var body: some View {
        VStack {
            // Experiment selector
            ExperimentSelectorView(selectedExperiments: $selectedExperiments)
            
            // Metric selector
            MetricSelectorView(
                availableMetrics: commonMetrics,
                selectedMetrics: $selectedMetrics,
                searchFilter: .constant("")
            )
            
            // Comparison chart
            ComparisonChartView(
                experiments: selectedExperiments,
                selectedMetrics: selectedMetrics
            )
        }
    }
}

struct ComparisonChartView: View {
    let experiments: [Experiment]
    let selectedMetrics: [String]
    
    var chartType: ChartType {
        switch selectedMetrics.count {
        case 1: return .bar
        case 2: return .scatter
        case 3...: return .radar
        default: return .empty
        }
    }
    
    var body: some View {
        Group {
            switch chartType {
            case .bar:
                BarComparisonChart(experiments: experiments, metric: selectedMetrics.first!)
            case .scatter:
                ScatterComparisonChart(experiments: experiments, metrics: Array(selectedMetrics.prefix(2)))
            case .radar:
                RadarComparisonChart(experiments: experiments, metrics: selectedMetrics)
            case .empty:
                Text("Select metrics to compare")
            }
        }
        .frame(height: 400)
    }
}

enum ChartType {
    case bar, scatter, radar, empty
}
```

## Service Layer Architecture

### 1. Authentication Manager
```swift
class AuthenticationManager: ObservableObject {
    @Published var isAuthenticated = false
    @Published var currentUser: User?
    
    private let supabaseClient: SupabaseClient
    
    init() {
        self.supabaseClient = SupabaseClient(
            supabaseURL: URL(string: "YOUR_SUPABASE_URL")!,
            supabaseKey: "YOUR_SUPABASE_ANON_KEY"
        )
    }
    
    func login(email: String, password: String) async {
        // Implement Supabase authentication
    }
    
    func logout() async {
        // Implement logout
    }
    
    func signup(email: String, password: String) async {
        // Implement signup
    }
}
```

### 2. API Client
```swift
class APIClient: ObservableObject {
    private let baseURL = "https://your-api.com"
    private let session = URLSession.shared
    
    func getWorkspaces() async throws -> [Workspace] {
        // Implement API call
    }
    
    func getExperiments(workspaceId: String) async throws -> [Experiment] {
        // Implement API call
    }
    
    func getExperimentMetrics(experimentId: String) async throws -> [Metric] {
        // Implement API call
    }
    
    func createExperiment(_ experiment: Experiment) async throws -> Experiment {
        // Implement API call
    }
    
    func updateExperiment(_ experiment: Experiment) async throws -> Experiment {
        // Implement API call
    }
    
    func createMetric(_ metric: Metric) async throws -> Metric {
        // Implement API call
    }
}
```

### 3. Data Manager
```swift
class DataManager: ObservableObject {
    @Published var workspaces: [Workspace] = []
    @Published var experiments: [Experiment] = []
    
    private let apiClient = APIClient()
    private let modelContext: ModelContext
    
    init(modelContext: ModelContext) {
        self.modelContext = modelContext
    }
    
    func loadWorkspaces() async {
        do {
            let workspaces = try await apiClient.getWorkspaces()
            await MainActor.run {
                self.workspaces = workspaces
            }
        } catch {
            // Handle error
        }
    }
    
    func loadExperiments(for workspace: Workspace) async {
        do {
            let experiments = try await apiClient.getExperiments(workspaceId: workspace.id)
            await MainActor.run {
                self.experiments = experiments
            }
        } catch {
            // Handle error
        }
    }
}
```

### 4. AI Analysis Service
```swift
class AIAnalysisService {
    private let anthropicAPIKey: String
    private let session = URLSession.shared
    
    init(apiKey: String) {
        self.anthropicAPIKey = apiKey
    }
    
    func analyzeExperiment(_ experiment: Experiment) async throws -> ExperimentAnalysis {
        let request = createAnalysisRequest(for: experiment)
        let (data, _) = try await session.data(for: request)
        return try JSONDecoder().decode(ExperimentAnalysis.self, from: data)
    }
    
    private func createAnalysisRequest(for experiment: Experiment) -> URLRequest {
        // Implement Claude API request
        var request = URLRequest(url: URL(string: "https://api.anthropic.com/v1/messages")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(anthropicAPIKey)", forHTTPHeaderField: "Authorization")
        
        let body = [
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1024,
            "messages": [
                [
                    "role": "user",
                    "content": createAnalysisPrompt(for: experiment)
                ]
            ]
        ]
        
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        return request
    }
    
    private func createAnalysisPrompt(for experiment: Experiment) -> String {
        // Create analysis prompt based on experiment data
        return """
        Analyze this ML experiment:
        
        Name: \(experiment.name)
        Description: \(experiment.description)
        Hyperparameters: \(experiment.hyperparams.map { "\($0.key): \($0.value)" }.joined(separator: ", "))
        Metrics: \(experiment.metrics.map { "\($0.name): \($0.value)" }.joined(separator: ", "))
        
        Provide insights and recommendations for optimization.
        """
    }
}

struct ExperimentAnalysis: Codable {
    let summary: String
    let insights: [String]
    let recommendations: [String]
    let hyperparameterRecommendations: [String: HPRecommendation]
}

struct HPRecommendation: Codable {
    let recommendation: String
    let level: Int
}
```

## Key Implementation Details

### 1. Chart Implementation with Swift Charts
- Use `Chart` view with `LineMark`, `BarMark`, `PointMark`
- Implement custom chart types for radar charts using `Path` and `Canvas`
- Add interactive features like zoom, pan, and metric selection
- Color coding for different metrics using chart foreground styles

### 2. Real-time Updates
- Use `URLSession` with WebSocket for real-time metric updates
- Implement `ObservableObject` pattern for reactive UI updates
- Background queue processing for metric calculations

### 3. Search and Filtering
- Implement `searchable` modifier for experiment and metric filtering
- Use `@State` for search text and computed properties for filtered results
- Debounced search to improve performance

### 4. Data Persistence
- Use SwiftData/Core Data for local caching
- Implement sync logic with remote API
- Handle offline scenarios with cached data

### 5. Export Functionality
- CSV export for metrics data
- PNG/PDF export for charts
- Share sheet integration for native sharing

### 6. Performance Optimizations
- Lazy loading for large experiment lists
- Virtualized charts for large datasets
- Background processing for metric calculations
- Image caching for chart thumbnails

This implementation plan provides a complete 1:1 functional port of the Tora web application to native macOS using SwiftUI, maintaining all existing features while leveraging native macOS capabilities.