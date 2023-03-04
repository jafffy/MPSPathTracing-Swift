//
//  Renderer+ComputeUtils.swift
//  MPSPathTracing-Swift
//
//  Created by Jaewon Choi on 2023/03/03.
//

import Metal
import MetalKit

extension Renderer {

    func buildComputePipelinesWithDevice(device: MTLDevice,
                                              metalKitView: MTKView) {
        let library = device.makeDefaultLibrary()!

        let computeDescriptor = MTLComputePipelineDescriptor()
        computeDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true

        computeDescriptor.computeFunction = library.makeFunction(name: "rayKernel")
        rayPipeline = try! device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)

        computeDescriptor.computeFunction = library.makeFunction(name: "shadeKernel")
        shadePipeline = try! device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)

        computeDescriptor.computeFunction = library.makeFunction(name: "shadowKernel")
        shadowPipeline = try! device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)

        computeDescriptor.computeFunction = library.makeFunction(name: "accumulateKernel")
        accumulatePipeline = try! device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
    }
}
