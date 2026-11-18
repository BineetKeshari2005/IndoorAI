"""Site-Specific Wi-Fi Radio Map & Fingerprint Localization Module."""

from src.wifi_localization.evaluator import WifiLocalizationEvaluator
from src.wifi_localization.knn_localizer import KnnLocalizer
from src.wifi_localization.models import (
    ApObservation,
    ApStatistics,
    CandidateMatch,
    EvaluationMetrics,
    LocalizationPrediction,
    LocationRadioProfile,
    RadioMap,
    RadioMapMetadata,
    ReferenceLocation,
    SurveySweep,
)
from src.wifi_localization.normalization import (
    clean_ap_observation,
    classify_channel_band,
    deduplicate_sweep_observations,
    is_valid_rssi,
    normalize_bssid,
)
from src.wifi_localization.probabilistic_localizer import ProbabilisticLocalizer
from src.wifi_localization.radio_map_builder import RadioMapBuilder
from src.wifi_localization.storage import (
    delete_radio_map,
    get_radio_map_dir,
    list_radio_maps,
    load_radio_map,
    load_reference_locations,
    save_radio_map,
    save_reference_locations,
)

__all__ = [
    "ApObservation",
    "ApStatistics",
    "CandidateMatch",
    "EvaluationMetrics",
    "LocalizationPrediction",
    "LocationRadioProfile",
    "RadioMap",
    "RadioMapMetadata",
    "ReferenceLocation",
    "SurveySweep",
    "normalize_bssid",
    "is_valid_rssi",
    "classify_channel_band",
    "clean_ap_observation",
    "deduplicate_sweep_observations",
    "RadioMapBuilder",
    "KnnLocalizer",
    "ProbabilisticLocalizer",
    "WifiLocalizationEvaluator",
    "get_radio_map_dir",
    "save_radio_map",
    "load_radio_map",
    "list_radio_maps",
    "delete_radio_map",
    "save_reference_locations",
    "load_reference_locations",
]
