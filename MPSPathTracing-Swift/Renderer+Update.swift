//
//  Renderer+Update.swift
//  MPSPathTracing-Swift
//
//  Created by Jaewon Choi on 2023/03/03.
//

import Metal
import MetalKit

extension Renderer {

    func update() {
        /// Update any game state before rendering

        uniforms[0].projectionMatrix = projectionMatrix

        let rotationAxis = SIMD3<Float>(1, 1, 0)
        let modelMatrix = matrix4x4_rotation(radians: rotation, axis: rotationAxis)
        let viewMatrix = matrix4x4_translation(0.0, 0.0, -8.0)
        uniforms[0].modelViewMatrix = simd_mul(viewMatrix, modelMatrix)
    }

    private func updateDynamicBufferState() {
        /// Update the state of our uniform buffers before rendering

        uniformBufferIndex = (uniformBufferIndex + 1) % maxBuffersInFlight

        uniformBufferOffset = alignedUniformsSize * uniformBufferIndex

        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset).bindMemory(to:Uniforms.self, capacity:1)
    }

    func draw(in view: MTKView) {
        /// Per frame updates hare

        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)

        if let commandBuffer = commandQueue.makeCommandBuffer() {

            let semaphore = inFlightSemaphore
            commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
                semaphore.signal()
            }

            self.updateDynamicBufferState()

            self.update()

            /// Delay getting the currentRenderPassDescriptor until we absolutely need it to avoid
            ///   holding onto the drawable and blocking the display pipeline any longer than necessary
            let renderPassDescriptor = view.currentRenderPassDescriptor

            if let renderPassDescriptor = renderPassDescriptor, let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {

                /// Final pass rendering code here
                renderEncoder.label = "Primary Render Encoder"

                renderEncoder.pushDebugGroup("Draw Box")

                renderEncoder.setCullMode(.back)

                renderEncoder.setFrontFacing(.counterClockwise)

                renderEncoder.setRenderPipelineState(pipelineState)

                renderEncoder.setDepthStencilState(depthState)

                renderEncoder.setVertexBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
                renderEncoder.setFragmentBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)

                for (index, element) in mesh.vertexDescriptor.layouts.enumerated() {
                    guard let layout = element as? MDLVertexBufferLayout else {
                        return
                    }

                    if layout.stride != 0 {
                        let buffer = mesh.vertexBuffers[index]
                        renderEncoder.setVertexBuffer(buffer.buffer, offset:buffer.offset, index: index)
                    }
                }

                renderEncoder.setFragmentTexture(colorMap, index: TextureIndex.color.rawValue)

                for submesh in mesh.submeshes {
                    renderEncoder.drawIndexedPrimitives(type: submesh.primitiveType,
                                                        indexCount: submesh.indexCount,
                                                        indexType: submesh.indexType,
                                                        indexBuffer: submesh.indexBuffer.buffer,
                                                        indexBufferOffset: submesh.indexBuffer.offset)

                }

                renderEncoder.popDebugGroup()

                renderEncoder.endEncoding()

                if let drawable = view.currentDrawable {
                    commandBuffer.present(drawable)
                }
            }

            commandBuffer.commit()
        }
    }
}
