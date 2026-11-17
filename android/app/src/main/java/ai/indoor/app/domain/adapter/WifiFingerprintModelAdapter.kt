package ai.indoor.app.domain.adapter

import ai.indoor.app.domain.models.WifiFingerprint

/**
 * Exception thrown when attempting to feed an uncalibrated real-world Wi-Fi fingerprint
 * into models expecting the UJIIndoorLoc 520-WAP feature space without calibration.
 */
class UncalibratedFingerprintException(message: String) : UnsupportedOperationException(message)

/**
 * Result of attempting to adapt a real-world or synthetic fingerprint for ML model inference.
 */
sealed class ModelAdaptationResult {
    /**
     * Successful mapping using a verified empirical survey calibration map.
     */
    data class Calibrated(
        val wapFeatureVector: FloatArray,
        val matchedWapCount: Int,
        val totalExpectedWaps: Int = 520
    ) : ModelAdaptationResult() {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as Calibrated
            return wapFeatureVector.contentEquals(other.wapFeatureVector) &&
                    matchedWapCount == other.matchedWapCount &&
                    totalExpectedWaps == other.totalExpectedWaps
        }

        override fun hashCode(): Int {
            var result = wapFeatureVector.contentHashCode()
            result = 31 * result + matchedWapCount
            result = 31 * result + totalExpectedWaps
            return result
        }
    }

    /**
     * Uncalibrated real-world fingerprint. Direct inference is refused to avoid invalid positioning.
     */
    data class Uncalibrated(
        val reason: String,
        val observedBssidCount: Int
    ) : ModelAdaptationResult()

    /**
     * Synthetic benchmark fingerprint strictly used for demonstration/testing.
     */
    data class DemoSynthetic(
        val simulatedBuildingId: Int,
        val simulatedFloor: Int
    ) : ModelAdaptationResult()
}

/**
 * Architectural adapter boundary preventing arbitrary or fabricated mapping between
 * real-world 802.11 BSSIDs and the UJIIndoorLoc dataset's abstract WAP001–WAP520 feature columns.
 *
 * CRITICAL REQUIREMENT (Task 14):
 * Existing models (Random Forest, k-NN, Multitask MLP) expect 520 abstract WAP features.
 * Real Android scans return hardware BSSIDs. Real BSSIDs MUST NOT be mapped to WAP columns
 * arbitrarily or fabricated. An empirical radio survey calibration mapping is strictly
 * required before live inference can be enabled.
 */
class WifiFingerprintModelAdapter(
    private val calibrationMapping: Map<String, String>? = null
) {
    val isCalibrated: Boolean
        get() = !calibrationMapping.isNullOrEmpty()

    /**
     * Evaluates whether a fingerprint can be adapted for UJIIndoorLoc model inference.
     */
    fun adaptForInference(fingerprint: WifiFingerprint): ModelAdaptationResult {
        if (fingerprint.isMock) {
            return ModelAdaptationResult.DemoSynthetic(
                simulatedBuildingId = 0,
                simulatedFloor = 1
            )
        }

        if (calibrationMapping.isNullOrEmpty()) {
            return ModelAdaptationResult.Uncalibrated(
                reason = "Real-world Wi-Fi scans observe local hardware BSSIDs. The existing ML models (Random Forest, k-NN, Multitask MLP) were trained on UJI campus features (WAP001–WAP520). A site survey calibration mapping is required before live localization inference can be performed.",
                observedBssidCount = fingerprint.count
            )
        }

        // Calibrated empirical survey mapping: Map observed BSSID to WAP index
        val vector = FloatArray(520) { -105.0f } // Default noise floor (-105 dBm)
        var matched = 0

        fingerprint.accessPoints.forEach { ap ->
            val wapId = calibrationMapping[ap.bssid.lowercase()]
            if (wapId != null && wapId.startsWith("WAP")) {
                val index = wapId.removePrefix("WAP").toIntOrNull()?.minus(1)
                if (index != null && index in 0 until 520) {
                    vector[index] = ap.rssi.toFloat()
                    matched++
                }
            }
        }

        return ModelAdaptationResult.Calibrated(
            wapFeatureVector = vector,
            matchedWapCount = matched
        )
    }
}
