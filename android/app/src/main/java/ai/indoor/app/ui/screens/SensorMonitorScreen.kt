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
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CleaningServices
import androidx.compose.material.icons.filled.FileDownload
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
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
import ai.indoor.app.domain.models.SensorSamplingRate
import ai.indoor.app.ui.components.MockDataBadge
import ai.indoor.app.ui.components.OrientationReadingCard
import ai.indoor.app.ui.components.SensorReadingCard
import ai.indoor.app.ui.theme.SignalGreen
import ai.indoor.app.ui.theme.SignalRed
import ai.indoor.app.viewmodel.SensorViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SensorMonitorScreen(
    viewModel: SensorViewModel,
    modifier: Modifier = Modifier
) {
    val readings by viewModel.readings.collectAsState()
    val isListening by viewModel.isListening.collectAsState()
    val samplingRate by viewModel.samplingRate.collectAsState()
    val exportPreview by viewModel.exportPreview.collectAsState()
    val exportFormat by viewModel.exportFormat.collectAsState()

    val availability = viewModel.availability
    val scrollState = rememberScrollState()
    var samplingMenuExpanded by remember { mutableStateOf(false) }

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
                    text = "Inertial & Magnetic Sensors",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Real-time user-controlled smartphone sensor streamer",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            if (availability.isMock) {
                MockDataBadge()
            } else {
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(6.dp))
                        .background(Color(0xFFDCFCE7))
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text(
                        text = "HARDWARE ACTIVE",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF15803D)
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Control & Configuration Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
        ) {
            Column(modifier = Modifier.padding(14.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Sampling Control",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = if (isListening) "Sampling active (${samplingRate.label})" else "Sensors idle (Zero battery drain)",
                            style = MaterialTheme.typography.bodySmall,
                            color = if (isListening) SignalGreen else MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }

                    Button(
                        onClick = { viewModel.toggleListening() },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (isListening) SignalRed else MaterialTheme.colorScheme.primary
                        ),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Icon(
                            imageVector = if (isListening) Icons.Default.Stop else Icons.Default.PlayArrow,
                            contentDescription = null
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(if (isListening) "Stop" else "Start")
                    }
                }

                Spacer(modifier = Modifier.height(10.dp))

                // Configurable Sampling Rate Dropdown
                ExposedDropdownMenuBox(
                    expanded = samplingMenuExpanded,
                    onExpandedChange = { samplingMenuExpanded = it }
                ) {
                    OutlinedTextField(
                        value = "${samplingRate.label} — ${samplingRate.rateDescription}",
                        onValueChange = {},
                        readOnly = true,
                        label = { Text("Sampling Rate Profile") },
                        trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = samplingMenuExpanded) },
                        modifier = Modifier
                            .menuAnchor()
                            .fillMaxWidth()
                    )

                    ExposedDropdownMenu(
                        expanded = samplingMenuExpanded,
                        onDismissRequest = { samplingMenuExpanded = false }
                    ) {
                        SensorSamplingRate.values().forEach { rate ->
                            DropdownMenuItem(
                                text = {
                                    Column {
                                        Text(text = rate.label, fontWeight = FontWeight.SemiBold)
                                        Text(text = rate.rateDescription, style = MaterialTheme.typography.labelSmall, color = Color.Gray)
                                    }
                                },
                                onClick = {
                                    viewModel.setSamplingRate(rate)
                                    samplingMenuExpanded = false
                                }
                            )
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(14.dp))

        // Sensor 1: Accelerometer
        SensorReadingCard(
            sensorName = "Tri-Axial Accelerometer",
            unit = "m/s²",
            x = readings.ax,
            y = readings.ay,
            z = readings.az,
            magnitude = readings.accelerationMagnitude,
            isAvailable = availability.hasAccelerometer,
            accuracy = readings.accelAccuracy,
            timestampNs = readings.accelTimestampNs,
            extraMetrics = {
                Column {
                    Text(
                        text = "Low-Pass Gravity Estimate (α=0.8): [X: %+.2f, Y: %+.2f, Z: %+.2f] m/s²".format(
                            readings.gravityX, readings.gravityY, readings.gravityZ
                        ),
                        style = MaterialTheme.typography.labelSmall,
                        fontFamily = FontFamily.Monospace,
                        color = MaterialTheme.colorScheme.secondary
                    )
                    Text(
                        text = "Dynamic Linear Acceleration Magnitude: %.2f m/s²".format(readings.linearAccelerationMagnitude),
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        )

        Spacer(modifier = Modifier.height(10.dp))

        // Sensor 2: Gyroscope
        SensorReadingCard(
            sensorName = "Tri-Axial Gyroscope",
            unit = "rad/s",
            x = readings.gx,
            y = readings.gy,
            z = readings.gz,
            magnitude = readings.angularSpeedMagnitude,
            isAvailable = availability.hasGyroscope,
            accuracy = readings.gyroAccuracy,
            timestampNs = readings.gyroTimestampNs
        )

        Spacer(modifier = Modifier.height(10.dp))

        // Sensor 3: Magnetometer
        SensorReadingCard(
            sensorName = "Tri-Axial Magnetometer (Compass)",
            unit = "µT",
            x = readings.mx,
            y = readings.my,
            z = readings.mz,
            magnitude = readings.magneticFieldMagnitude,
            isAvailable = availability.hasMagnetometer,
            accuracy = readings.magAccuracy,
            timestampNs = readings.magTimestampNs,
            extraMetrics = {
                Text(
                    text = "Notice: Indoor steel structural elements, conduits, and equipment induce magnetic distortions. Raw magnetometer readings require spatial calibration.",
                    style = MaterialTheme.typography.labelSmall,
                    color = Color(0xFFB45309)
                )
            }
        )

        Spacer(modifier = Modifier.height(10.dp))

        // Sensor 4: Orientation Foundation (Triad Fusion)
        OrientationReadingCard(
            azimuthDeg = readings.azimuthDeg,
            pitchDeg = readings.pitchDeg,
            rollDeg = readings.rollDeg,
            hasOrientation = readings.hasOrientation
        )

        Spacer(modifier = Modifier.height(14.dp))

        // Local Storage & Export Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
            )
        ) {
            Column(modifier = Modifier.padding(14.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.FileDownload,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = "Local Telemetry Export",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.weight(1f))
                    Text(
                        text = "Buffered: ${viewModel.getRecordedCount()} records",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.primary
                    )
                }

                Spacer(modifier = Modifier.height(10.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Button(
                        onClick = { viewModel.exportAsCsv() },
                        enabled = viewModel.getRecordedCount() > 0,
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("Export CSV")
                    }

                    OutlinedButton(
                        onClick = { viewModel.exportAsJson() },
                        enabled = viewModel.getRecordedCount() > 0,
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("Export JSON")
                    }

                    OutlinedButton(
                        onClick = { viewModel.clearCollectedReadings() },
                        enabled = viewModel.getRecordedCount() > 0,
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Icon(imageVector = Icons.Default.CleaningServices, contentDescription = "Clear")
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(14.dp))

        // Sensor Physics & Privacy Information Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f)
            )
        ) {
            Row(
                modifier = Modifier.padding(12.dp),
                verticalAlignment = Alignment.Top
            ) {
                Icon(
                    imageVector = Icons.Default.Info,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(top = 2.dp)
                )
                Spacer(modifier = Modifier.width(10.dp))
                Column {
                    Text(
                        text = "Sensor Physics & Anti-Drift Guarantees",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = "• Accelerometer double-integration drifts quadratically O(t²) and cannot provide position alone.\n• Gyroscope integration drifts over time without absolute reference.\n• Magnetometers suffer from indoor hard/soft iron distortions.\n• Foreground sampling only: zero background collection.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))
    }

    // Export Preview Modal Dialog
    if (exportPreview != null) {
        AlertDialog(
            onDismissRequest = { viewModel.clearExportPreview() },
            title = {
                Text("Exported Sensor Telemetry ($exportFormat)")
            },
            text = {
                Column {
                    Text(
                        text = "Hardware nanosecond timestamps preserved for differential physics:",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    SelectionContainer {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .heightIn(max = 300.dp)
                                .clip(RoundedCornerShape(6.dp))
                                .background(MaterialTheme.colorScheme.surfaceVariant)
                                .padding(10.dp)
                                .verticalScroll(rememberScrollState())
                        ) {
                            Text(
                                text = exportPreview!!,
                                fontFamily = FontFamily.Monospace,
                                fontSize = 11.sp,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            },
            confirmButton = {
                Button(onClick = { viewModel.clearExportPreview() }) {
                    Text("Close")
                }
            }
        )
    }
}
