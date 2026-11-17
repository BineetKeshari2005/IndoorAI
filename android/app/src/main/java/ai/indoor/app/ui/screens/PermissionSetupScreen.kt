package ai.indoor.app.ui.screens

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.Sensors
import androidx.compose.material.icons.filled.Wifi
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import ai.indoor.app.ui.components.PermissionCard

@Composable
fun PermissionSetupScreen(
    onNavigateBack: () -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val scrollState = rememberScrollState()

    fun checkPerm(perm: String): Boolean =
        ContextCompat.checkSelfPermission(context, perm) == PackageManager.PERMISSION_GRANTED

    var hasLocation by remember { mutableStateOf(checkPerm(Manifest.permission.ACCESS_FINE_LOCATION)) }
    var hasCamera by remember { mutableStateOf(checkPerm(Manifest.permission.CAMERA)) }

    val locationLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestMultiplePermissions()
    ) { perms ->
        hasLocation = perms[Manifest.permission.ACCESS_FINE_LOCATION] == true ||
                perms[Manifest.permission.ACCESS_COARSE_LOCATION] == true
    }

    val cameraLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCamera = granted
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(16.dp)
    ) {
        Text(
            text = "Permissions & Privacy Architecture",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold
        )

        Spacer(modifier = Modifier.height(6.dp))

        Text(
            text = "IndoorAI requires specific Android permissions to inspect local Wi-Fi radio beacons and sensor telemetry. Permissions are requested only when features are activated.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(16.dp))

        // Card 1: Location for Wi-Fi Scanning
        PermissionCard(
            title = "Fine Location Permission",
            description = "Allows the app to trigger Wi-Fi network sweeps and receive nearby access point signal strengths (BSSID, RSSI, channel frequency).",
            whyNeeded = "Android OS treats Wi-Fi beacon signals as spatial identifiers. Since Android 9 (API 28), calling WifiManager.getScanResults() requires ACCESS_FINE_LOCATION.",
            icon = Icons.Default.LocationOn,
            isGranted = hasLocation,
            onRequestPermission = {
                locationLauncher.launch(
                    arrayOf(
                        Manifest.permission.ACCESS_FINE_LOCATION,
                        Manifest.permission.ACCESS_COARSE_LOCATION
                    )
                )
            }
        )

        Spacer(modifier = Modifier.height(12.dp))

        // Card 2: Wi-Fi State
        PermissionCard(
            title = "Wi-Fi State & Radio Access",
            description = "Required to inspect whether Wi-Fi is enabled and request radio sweeps via WifiManager.startScan().",
            whyNeeded = "Normal permission granted automatically at installation via AndroidManifest.xml (ACCESS_WIFI_STATE and CHANGE_WIFI_STATE).",
            icon = Icons.Default.Wifi,
            isGranted = true,
            onRequestPermission = {}
        )

        Spacer(modifier = Modifier.height(12.dp))

        // Card 3: Camera
        PermissionCard(
            title = "Camera (Manual Landmark Capture)",
            description = "Required ONLY when you explicitly press the 'Capture Photograph' button to take a picture of an architectural landmark or sign.",
            whyNeeded = "Camera access is never initiated in the background. Images are stripped of EXIF tags and blurred client-side before transmission.",
            icon = Icons.Default.CameraAlt,
            isGranted = hasCamera,
            onRequestPermission = {
                cameraLauncher.launch(Manifest.permission.CAMERA)
            }
        )

        Spacer(modifier = Modifier.height(12.dp))

        // Card 4: Motion Sensors
        PermissionCard(
            title = "Motion & Orientation Sensors",
            description = "Reads phone accelerometer, gyroscope, and magnetometer to support Pedestrian Dead Reckoning (PDR) step estimation.",
            whyNeeded = "Standard on-board sensors do not require runtime prompts on Android < 12. HIGH_SAMPLING_RATE_SENSORS is declared in manifest for Android 12+.",
            icon = Icons.Default.Sensors,
            isGranted = true,
            onRequestPermission = {}
        )

        Spacer(modifier = Modifier.height(20.dp))

        // Philosophy Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            Column(modifier = Modifier.padding(14.dp)) {
                Text(
                    text = "Anti-Surveillance Guarantee",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "• No background camera or microphone execution.\n• No collection of device IMEI, phone numbers, or Google accounts.\n• Ephemeral, daily-rotated device tokens (HMAC-SHA256).\n• You can revoke permissions or stop capture sessions at any moment.",
                    style = MaterialTheme.typography.bodySmall
                )
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        Button(
            onClick = onNavigateBack,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Done / Return Home")
        }
    }
}

