package ai.indoor.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.navigation.compose.rememberNavController
import ai.indoor.app.data.repository.LocalizationRepositoryImpl
import ai.indoor.app.domain.usecases.ManageCaptureSessionUseCase
import ai.indoor.app.sensors.AndroidSensorProvider
import ai.indoor.app.sensors.MockSensorProvider
import ai.indoor.app.ui.navigation.AppNavHost
import ai.indoor.app.ui.theme.IndoorAITheme
import ai.indoor.app.viewmodel.ImageCaptureViewModel
import ai.indoor.app.viewmodel.MainViewModel
import ai.indoor.app.viewmodel.SensorViewModel
import ai.indoor.app.viewmodel.SessionViewModel
import ai.indoor.app.viewmodel.WifiScanViewModel
import ai.indoor.app.wifi.AndroidWifiScanner
import ai.indoor.app.wifi.MockWifiScanner

class MainActivity : ComponentActivity() {

    private val repository by lazy { LocalizationRepositoryImpl() }
    private val sessionUseCase by lazy { ManageCaptureSessionUseCase(repository) }

    // Providers: use Android hardware where available, fallback gracefully to Mock
    private val wifiScanner by lazy {
        try {
            AndroidWifiScanner(applicationContext)
        } catch (_: Exception) {
            MockWifiScanner()
        }
    }

    private val sensorProvider by lazy {
        try {
            val provider = AndroidSensorProvider(applicationContext)
            if (provider.availability.hasAnySensor) provider else MockSensorProvider()
        } catch (_: Exception) {
            MockSensorProvider()
        }
    }

    private val photoStorageManager by lazy {
        ai.indoor.app.camera.storage.PhotoStorageManager(applicationContext)
    }

    private val cameraManager by lazy {
        try {
            val mgr = ai.indoor.app.camera.AndroidCameraXManager(applicationContext)
            if (mgr.isCameraAvailable()) mgr else ai.indoor.app.camera.MockCameraManager(applicationContext)
        } catch (_: Exception) {
            ai.indoor.app.camera.MockCameraManager(applicationContext)
        }
    }

    private val mainViewModel: MainViewModel by viewModels {
        object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(modelClass: Class<T>): T {
                return MainViewModel(repository) as T
            }
        }
    }

    private val wifiScanViewModel: WifiScanViewModel by viewModels {
        object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(modelClass: Class<T>): T {
                return WifiScanViewModel(scanner = wifiScanner, repository = repository) as T
            }
        }
    }

    private val sensorViewModel: SensorViewModel by viewModels {
        object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(modelClass: Class<T>): T {
                return SensorViewModel(provider = sensorProvider) as T
            }
        }
    }

    private val imageCaptureViewModel: ImageCaptureViewModel by viewModels {
        object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(modelClass: Class<T>): T {
                return ImageCaptureViewModel(
                    cameraManager = cameraManager,
                    storageManager = photoStorageManager,
                    sessionUseCase = sessionUseCase
                ) as T
            }
        }
    }

    private val sessionViewModel: SessionViewModel by viewModels {
        object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(modelClass: Class<T>): T {
                return SessionViewModel(sessionUseCase, sensorProvider) as T
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            IndoorAITheme {
                val navController = rememberNavController()
                AppNavHost(
                    navController = navController,
                    mainViewModel = mainViewModel,
                    wifiScanViewModel = wifiScanViewModel,
                    sensorViewModel = sensorViewModel,
                    imageCaptureViewModel = imageCaptureViewModel,
                    sessionViewModel = sessionViewModel
                )
            }
        }
    }

    override fun onStop() {
        super.onStop()
        // Ensure motion and magnetic sensors are stopped when activity moves to background
        sensorProvider.stopListening()
    }
}
