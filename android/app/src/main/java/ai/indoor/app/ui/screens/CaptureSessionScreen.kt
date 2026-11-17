package ai.indoor.app.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddCircle
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.CloudSync
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Sensors
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Wifi
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.indoor.app.domain.models.CaptureSession
import ai.indoor.app.ui.components.MockDataBadge
import ai.indoor.app.ui.components.StatusBadge
import ai.indoor.app.ui.theme.SignalGreen
import ai.indoor.app.ui.theme.SignalRed
import ai.indoor.app.viewmodel.MainViewModel
import ai.indoor.app.viewmodel.SessionViewModel

@Composable
fun CaptureSessionScreen(
    sessionViewModel: SessionViewModel,
    mainViewModel: MainViewModel,
    modifier: Modifier = Modifier
) {
    val currentSession by sessionViewModel.currentSession.collectAsState()
    val elapsedSeconds by sessionViewModel.elapsedSeconds.collectAsState()
    val syncMessage by sessionViewModel.syncMessage.collectAsState()
    val backendUrl by mainViewModel.backendUrl.collectAsState()
    val isServerOnline by mainViewModel.isServerOnline.collectAsState()
    val streamSensors by sessionViewModel.streamSensors.collectAsState()

    var urlInput by remember(backendUrl) { mutableStateOf(backendUrl) }
    val scrollState = rememberScrollState()

    val isRunning = currentSession?.isRunning == true

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(16.dp)
    ) {
        // Header
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Capture Session",
                    style = MaterialTheme.typography.headlineMedium
                )
                Text(
                    text = "Multimodal survey session recording",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            MockDataBadge()
        }

        Spacer(modifier = Modifier.height(14.dp))

        // Timer Display Hero Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(14.dp),
            colors = CardDefaults.cardColors(
                containerColor = if (isRunning) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                StatusBadge(
                    text = if (isRunning) "SESSION ACTIVE" else if (currentSession != null) "SESSION PAUSED" else "READY TO START",
                    backgroundColor = if (isRunning) SignalGreen.copy(alpha = 0.2f) else MaterialTheme.colorScheme.surface,
                    contentColor = if (isRunning) SignalGreen else MaterialTheme.colorScheme.onSurfaceVariant
                )

                Spacer(modifier = Modifier.height(12.dp))

                val mins = elapsedSeconds / 60
                val secs = elapsedSeconds % 60
                Text(
                    text = "%02d:%02d".format(mins, secs),
                    style = MaterialTheme.typography.headlineLarge,
                    fontSize = 44.sp,
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.ExtraBold,
                    color = if (isRunning) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface
                )

                Text(
                    text = currentSession?.sessionId ?: "No Active Session ID",
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                Spacer(modifier = Modifier.height(12.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "Stream IMU Sensors",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                    androidx.compose.material3.Switch(
                        checked = streamSensors,
                        onCheckedChange = { sessionViewModel.toggleSensorStreaming(it) }
                    )
                }

                Spacer(modifier = Modifier.height(12.dp))

                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    if (!isRunning) {
                        Button(
                            onClick = { sessionViewModel.startSession() },
                            modifier = Modifier.weight(1f).height(48.dp),
                            shape = RoundedCornerShape(10.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                        ) {
                            Icon(imageVector = Icons.Default.PlayArrow, contentDescription = null)
                            Spacer(modifier = Modifier.width(6.dp))
                            Text("Start Session")
                        }
                    } else {
                        Button(
                            onClick = { sessionViewModel.stopSession() },
                            modifier = Modifier.weight(1f).height(48.dp),
                            shape = RoundedCornerShape(10.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = SignalRed)
                        ) {
                            Icon(imageVector = Icons.Default.Stop, contentDescription = null)
                            Spacer(modifier = Modifier.width(6.dp))
                            Text("Stop Session")
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Observation Metrics Counters
        Text(
            text = "Collected Telemetry Counters",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold
        )

        Spacer(modifier = Modifier.height(8.dp))

        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            MetricCard(
                title = "Wi-Fi Scans",
                count = currentSession?.wifiScanCount ?: 0,
                icon = Icons.Default.Wifi,
                modifier = Modifier.weight(1f)
            )
            MetricCard(
                title = "IMU Samples",
                count = currentSession?.imuSampleCount ?: 0,
                icon = Icons.Default.Sensors,
                modifier = Modifier.weight(1f)
            )
            MetricCard(
                title = "Photos",
                count = currentSession?.photoCount ?: 0,
                icon = Icons.Default.CameraAlt,
                modifier = Modifier.weight(1f)
            )
        }

        if (isRunning) {
            Spacer(modifier = Modifier.height(12.dp))
            OutlinedButton(
                onClick = { sessionViewModel.addMockObservation() },
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(imageVector = Icons.Default.AddCircle, contentDescription = null)
                Spacer(modifier = Modifier.width(6.dp))
                Text("Simulate Ambient Telemetry Packet")
            }
        }

        Spacer(modifier = Modifier.height(18.dp))

        // Sync & Backend Section
        Text(
            text = "Backend Server Synchronization",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold
        )

        Spacer(modifier = Modifier.height(8.dp))

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
        ) {
            Column(modifier = Modifier.padding(14.dp)) {
                OutlinedTextField(
                    value = urlInput,
                    onValueChange = { urlInput = it },
                    label = { Text("Backend Base URL") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true
                )

                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "• Emulator Host: http://10.0.2.2:8000/\n• Physical Device: http://<YOUR_LAN_IP>:8000/",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.secondary
                )

                Spacer(modifier = Modifier.height(10.dp))

                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(
                        onClick = { mainViewModel.updateBackendUrl(urlInput) },
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Apply URL")
                    }

                    Button(
                        onClick = { sessionViewModel.syncWithBackend() },
                        enabled = currentSession != null,
                        modifier = Modifier.weight(1.2f)
                    ) {
                        Icon(imageVector = Icons.Default.CloudSync, contentDescription = null)
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Sync Session")
                    }
                }

                syncMessage?.let { msg ->
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = msg,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))
    }
}

@Composable
private fun MetricCard(
    title: String,
    count: Int,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(10.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
    ) {
        Column(
            modifier = Modifier.padding(10.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(20.dp)
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = count.toString(),
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = title,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

