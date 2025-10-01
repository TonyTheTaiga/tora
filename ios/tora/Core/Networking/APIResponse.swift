import Foundation

struct APIResponse<T: Decodable>: Decodable {
    let status: Int
    let data: T?
}
