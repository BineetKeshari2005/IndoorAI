package ai.indoor.app.ui.navigation

import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import ai.indoor.app.ui.screens.CaptureSessionScreen
import ai.indoor.app.ui.screens.ImageCaptureScreen
import ai.indoor.app.ui.screens.PermissionSetupScreen
import ai.indoor.app.ui.screens.SensorMonitorScreen
import ai.indoor.app.ui.screens.WelcomeScreen
import ai.indoor.app.ui.screens.WifiScanScreen
import ai.indoor.app.viewmodel.ImageCaptureViewModel
import ai.indoor.app.viewmodel.MainViewModel
import ai.indoor.app.viewmodel.SensorViewModel
import ai.indoor.app.viewmodel.SessionViewModel
import ai.indoor.app.viewmodel.WifiScanViewModel

@Composable
fun AppNavHost(
    navController: NavHostController,
    mainViewModel: MainViewModel,
    wifiScanViewModel: WifiScanViewModel,
    sensorViewModel: SensorViewModel,
    imageCaptureViewModel: ImageCaptureViewModel,
    sessionViewModel: SessionViewModel,
    modifier: Modifier = Modifier
) {
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route
    val isServerOnline by mainViewModel.isServerOnline.collectAsState()

    Scaffold(
        bottomBar = {
            NavigationBar {
                Screen.bottomNavItems.forEach { screen ->
                    val isSelected = currentRoute == screen.route
                    NavigationBarItem(
                        selected = isSelected,
                        onClick = {
                            navController.navigate(screen.route) {
                                popUpTo(navController.graph.findStartDestination().id) {
                                    saveState = true
                                }
                                launchSingleTop = true
                                restoreState = true
                            }
                        },
                        icon = {
                            Icon(
                                imageVector = screen.icon,
                                contentDescription = screen.title
                            )
                        },
                        label = { Text(screen.title) }
                    )
                }
            }
        },
        modifier = modifier
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Screen.Welcome.route,
            modifier = Modifier.padding(innerPadding)
        ) {
            composable(Screen.Welcome.route) {
                WelcomeScreen(
                    onNavigateToWifi = { navController.navigate(Screen.WifiScan.route) },
                    onNavigateToSensors = { navController.navigate(Screen.Sensors.route) },
                    onNavigateToCamera = { navController.navigate(Screen.Camera.route) },
                    onNavigateToSession = { navController.navigate(Screen.Session.route) },
                    onNavigateToPermissions = { navController.navigate(Screen.Permissions.route) },
                    isServerOnline = isServerOnline
                )
            }

            composable(Screen.WifiScan.route) {
                WifiScanScreen(viewModel = wifiScanViewModel)
            }

            composable(Screen.Sensors.route) {
                SensorMonitorScreen(viewModel = sensorViewModel)
            }

            composable(Screen.Camera.route) {
                ImageCaptureScreen(
                    viewModel = imageCaptureViewModel,
                    sessionViewModel = sessionViewModel
                )
            }

            composable(Screen.Session.route) {
                CaptureSessionScreen(
                    sessionViewModel = sessionViewModel,
                    mainViewModel = mainViewModel
                )
            }

            composable(Screen.Permissions.route) {
                PermissionSetupScreen(
                    onNavigateBack = { navController.popBackStack() }
                )
            }
        }
    }
}

