import Foundation

struct Metric: Decodable, Identifiable, Equatable {
    struct Metadata: Decodable, Equatable {
        let type: String?
    }

    var id: Int
    var experimentId: String
    var name: String
    var value: Double
    var step: Int?
    var metadata: Metadata?
    var createdAt: Date

    enum CodingKeys: String, CodingKey {
        case id
        case experimentId = "experiment_id"
        case name
        case value
        case step
        case metadata
        case createdAt = "created_at"
    }
}

struct HyperParam: Codable, Equatable {
    let key: String
    let value: HyperParamValue

    enum HyperParamValue: Codable, Equatable {
        case string(String)
        case int(Int)
        case double(Double)
        case bool(Bool)

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()

            if let stringValue = try? container.decode(String.self) {
                self = .string(stringValue)
            } else if let intValue = try? container.decode(Int.self) {
                self = .int(intValue)
            } else if let doubleValue = try? container.decode(Double.self) {
                self = .double(doubleValue)
            } else if let boolValue = try? container.decode(Bool.self) {
                self = .bool(boolValue)
            } else {
                throw DecodingError.typeMismatch(
                    HyperParamValue.self,
                    DecodingError.Context(
                        codingPath: decoder.codingPath,
                        debugDescription: "Unsupported hyperparam value type"
                    )
                )
            }
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            switch self {
            case .string(let value):
                try container.encode(value)
            case .int(let value):
                try container.encode(value)
            case .double(let value):
                try container.encode(value)
            case .bool(let value):
                try container.encode(value)
            }
        }

        var displayValue: String {
            switch self {
            case .string(let value):
                return value
            case .int(let value):
                return String(value)
            case .double(let value):
                return String(value)
            case .bool(let value):
                return String(value)
            }
        }
    }
}

struct Experiment: Decodable, Identifiable, Equatable {
    var id: String
    var name: String
    var description: String?
    var hyperparams: [HyperParam]
    var tags: [String]
    var createdAt: Date
    var updatedAt: Date
    var workspaceId: String?
    var url: String

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case description
        case hyperparams
        case tags
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case workspaceId = "workspace_id"
        case url
    }
}
