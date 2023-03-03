//
//  Renderer+Update.swift
//  MPSPathTracing-Swift
//
//  Created by Jaewon Choi on 2023/03/03.
//

import Foundation

extension Renderer {

    func update() {
        /// Update any game state before rendering

        uniforms[0].projectionMatrix = projectionMatrix

        let rotationAxis = SIMD3<Float>(1, 1, 0)
        let modelMatrix = matrix4x4_rotation(radians: rotation, axis: rotationAxis)
        let viewMatrix = matrix4x4_translation(0.0, 0.0, -8.0)
        uniforms[0].modelViewMatrix = simd_mul(viewMatrix, modelMatrix)
    }
}
