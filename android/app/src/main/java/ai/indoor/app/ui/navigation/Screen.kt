package ai.indoor.app.ui.navigation

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Security
import androidx.compose.material.icons.filled.Sensors
import androidx.compose.material.icons.filled.Timeline
import androidx.compose.material.icons.filled.Wifi
import androidx.compose.ui.graphics.vector.ImageVector

/**
 * Destinations across the IndoorAI mobile application.
 */
sealed class Screen(
    val route: String,
    val title: String,
    val icon: ImageVector
) {
    object Welcome : Screen("welcome", "Home", Icons.Default.Home)
    object WifiScan : Screen("wifi_scan", "Wi-Fi", Icons.Default.Wifi)
    object Sensors : Screen("sensors", "Sensors", Icons.Default.Sensors)
    object Camera : Screen("camera", "Camera", Icons.Default.CameraAlt)
    object Session : Screen("session", "Session", Icons.Default.Timeline)
    object Permissions : Screen("permissions", "Privacy", Icons.Default.Security)

    companion object {
        val bottomNavItems = listOf(
            Welcome,
            WifiScan,
            Sensors,
            Camera,
            Session
        )
    }
}

