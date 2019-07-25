import NaturalLanguage
import UIKit
import Vision

/*  ML
https://www.tensorflow.org/swift
https://developer.apple.com/machine-learning/
 */

/*  AGENDA
1) Explore out of the box
2) Train custom mustache classifier
3) Test, talk about limitations
*/











// Sentiment Analysis

//let str = "That mustache is the shit!"
//let tagger = NLTagger(tagSchemes: [.sentimentScore])
//tagger.string = str
//
//let (sentiment, _) = tagger.tag(at: str.startIndex, unit: .paragraph, scheme: .sentimentScore)
//let score = Double(sentiment!.rawValue)!
//
//if score > 0.0 {
//    print("Way to be positive 👍🏻")
//} else {
//    print("That wasn't very nice 😢")
//}




// Word Embeddings
// help you find nearest neighbors, good for fuzzy search

//let wrongStr = "moustache"
//var corrected = wrongStr
//let embedding = NLEmbedding.wordEmbedding(for: .english)!
//
//if !embedding.contains(wrongStr) {
//    let textChecker = UITextChecker()
//    let misspelledRange = textChecker.rangeOfMisspelledWord(in: wrongStr, range: NSRange(0..<wrongStr.utf8.count), startingAt: 0, wrap: false, language: "en_US")
//    corrected = textChecker.guesses(forWordRange: misspelledRange, in: wrongStr, language: "en_US")!.first!
//}
//
//let neighbors = embedding.neighbors(for: corrected, maximumCount: 10).sorted { $0.1 < $1.1 }
//
//print("input was \(wrongStr)")
//print("corrected was \(corrected)")
//print("neighbors found: ")
//print(neighbors)





// Computer Vision

/* EXAMPLES
 https://www.youtube.com/watch?v=NlpS-DhayQA
 https://www.youtube.com/watch?v=yeS8TJwBAFs
 https://ai.googleblog.com/2018/10/acoustic-detection-of-humpback-whales.html
 */


// Image classification
// You get Apple's out of the box

let fileURL = Bundle.main.url(forResource: "tom-selleck", withExtension: "jpg")!
let handler = VNImageRequestHandler(url: fileURL, options: [:])
let request = VNClassifyImageRequest()
try! handler.perform([request])


 let observations = request.results as! [VNClassificationObservation]
print("🍤🥟🍣 got \(observations.count) observations")
for obs in observations {
    print("label: \(obs.identifier) | confidence: \(obs.confidence)")
}

