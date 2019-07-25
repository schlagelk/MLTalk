import AppKit
import Vision
import PlaygroundSupport

// Ignore this
PlaygroundPage.current.needsIndefiniteExecution = true
let modelURL = Bundle.main.url(forResource: "ImageClassifier", withExtension: "mlmodelc")!
let objectClassifier = try! VNCoreMLModel(for: MLModel(contentsOf: modelURL))

let url = Bundle.main.url(forResource: "lorax", withExtension: "png")!
let hdURL = Bundle.main.url(forResource: "hotdog", withExtension: "jpg")!

let testImage = NSImage(contentsOf: url)!.cgImage(forProposedRect: nil, context: nil, hints: nil)!
let hdImage = NSImage(contentsOf: hdURL)!.cgImage(forProposedRect: nil, context: nil, hints: nil)!


// 👀

// 🌭
//// It thinks a hot dog is a mustache?
let mustacheRequest = VNCoreMLRequest(model: objectClassifier, completionHandler: { request, error in
    print(request.results!)
    PlaygroundPage.current.finishExecution()
})
let handler = VNImageRequestHandler(ciImage: CIImage(cgImage: testImage))
try! handler.perform([mustacheRequest])


// Vision has face detection!

// 👨🏻
var imageToTry: CGImage = testImage
let ciImage = CIImage(cgImage: imageToTry)

func handleFaceDetection (request: VNRequest, error: Error?) {
    guard let observations = request.results as? [VNFaceObservation] else { return }
    if observations.count > 0 {
        print("found a face!")

        let mustacheRequest = VNCoreMLRequest(model: objectClassifier, completionHandler: { request, error in
            if let results = request.results as? [VNClassificationObservation], results.first?.identifier == "mustache" {
                print("🤠: \(results.first!.confidence)")
            }
            PlaygroundPage.current.finishExecution()
        })
        let handler = VNImageRequestHandler(ciImage: ciImage)
        try! handler.perform([mustacheRequest])
    } else {
        print("didn't find a face")
    }
    PlaygroundPage.current.finishExecution()
}






let faceDetectionRequest = VNDetectFaceRectanglesRequest(completionHandler:handleFaceDetection)
let faceHandler = VNImageRequestHandler(cgImage: imageToTry, options: [:])
try! faceHandler.perform([faceDetectionRequest])



