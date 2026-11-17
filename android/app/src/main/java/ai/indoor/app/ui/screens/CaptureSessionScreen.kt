package ai.indoor.app.ui.screens

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
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
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.AddCircle
import androidx.compose.material.icons.filled.Archive
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.CloudSync
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Sensors
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Timeline
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material.icons.filled.Wifi
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.indoor.app.domain.models.AnnotationType
import ai.indoor.app.domain.models.DataSourceType
import ai.indoor.app.domain.models.SurveySession
import ai.indoor.app.domain.models.SurveySessionStatus
import ai.indoor.app.domain.timeline.SessionTimelineItem
import ai.indoor.app.domain.timeline.TimelineObservationType
import ai.indoor.app.ui.components.StatusBadge
import ai.indoor.app.ui.theme.SignalGreen
import ai.indoor.app.ui.theme.SignalRed
import ai.indoor.app.ui.theme.SignalYellow
import ai.indoor.app.viewmodel.MainViewModel
import ai.indoor.app.viewmodel.SessionViewModel
import java.io.File

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CaptureSessionScreen(
    sessionViewModel: SessionViewModel,
    mainViewModel: MainViewModel,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val scrollState = rememberScrollState()

    val currentSession by sessionViewModel.currentSession.collectAsState()
    val allSessions by sessionViewModel.allSessions.collectAsState()
    val timeline by sessionViewModel.timeline.collectAsState()
    val elapsedSeconds by sessionViewModel.elapsedSeconds.collectAsState()
    val streamSensors by sessionViewModel.streamSensors.collectAsState()
    val statusMessage by sessionViewModel.statusMessage.collectAsState()
    val exportedZip by sessionViewModel.exportedZipFile.collectAsState()
    val isExporting by sessionViewModel.isExporting.collectAsState()

    val backendUrl by mainViewModel.backendUrl.collectAsState()
    var urlInput by remember(backendUrl) { mutableStateOf(backendUrl) }

    // Dialog States
    var showCreateDialog by remember { mutableStateOf(false) }
    var showAnnotationDialog by remember { mutableStateOf(false) }
    var showDeleteConfirmDialog by remember { mutableStateOf(false) }
    var inspectingItem by remember { mutableStateOf<SessionTimelineItem?>(null) }

    val status = currentSession?.status ?: SurveySessionStatus.CREATED
    val isActive = status == SurveySessionStatus.ACTIVE
    val isPaused = status == SurveySessionStatus.PAUSED

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(16.dp)
    ) {
        // --- 1. Header ---
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Multimodal Survey Session",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Unified Wi-Fi, IMU, camera & landmark recording",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            currentSession?.let { sess ->
                val badgeColor = when (sess.dataSourceType) {
                    DataSourceType.REAL -> SignalGreen
                    DataSourceType.MOCK -> SignalYellow
                    DataSourceType.HYBRID -> MaterialTheme.colorScheme.primary
                }
                StatusBadge(text = sess.dataSourceType.displayName, color = badgeColor)
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // --- 2. Interruption Recovery Alert Banner ---
        if (currentSession?.isInterrupted == true) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = SignalYellow.copy(alpha = 0.2f))
            ) {
                Row(
                    modifier = Modifier.padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(imageVector = Icons.Default.Warning, contentDescription = null, tint = SignalYellow)
                    Spacer(modifier = Modifier.width(10.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Interrupted Session Recovered",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = currentSession?.interruptionNote ?: "Session was recovered after application restart.",
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                    Button(onClick = { sessionViewModel.resumeSession() }) {
                        Text("Resume")
                    }
                }
            }
            Spacer(modifier = Modifier.height(12.dp))
        }

        // --- 3. Active Session Hero Card ---
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = if (isActive) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            Column(
                modifier = Modifier.padding(18.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    StatusBadge(
                        text = status.displayName.uppercase(),
                        backgroundColor = if (isActive) SignalGreen.copy(alpha = 0.25f) else MaterialTheme.colorScheme.surface,
                        contentColor = if (isActive) SignalGreen else MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    currentSession?.let { sess ->
                        Text(
                            text = sess.sessionId,
                            style = MaterialTheme.typography.labelSmall,
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.SemiBold
                        )
                    }
                }

                Spacer(modifier = Modifier.height(10.dp))

                val mins = elapsedSeconds / 60
                val secs = elapsedSeconds % 60
                Text(
                    text = "%02d:%02d".format(mins, secs),
                    style = MaterialTheme.typography.headlineLarge,
                    fontSize = 46.sp,
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.ExtraBold,
                    color = if (isActive) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface
                )

                Text(
                    text = currentSession?.sessionName ?: "No Session Loaded",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )

                currentSession?.let { sess ->
                    Text(
                        text = sess.displayLocation,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                Spacer(modifier = Modifier.height(12.dp))

                // IMU Streaming Switch
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "Stream IMU Sensors to Session",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                    Switch(
                        checked = streamSensors,
                        onCheckedChange = { sessionViewModel.toggleSensorStreaming(it) }
                    )
                }

                Spacer(modifier = Modifier.height(14.dp))

                // Lifecycle Controls Row
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    when (status) {
                        SurveySessionStatus.CREATED -> {
                            Button(
                                onClick = { sessionViewModel.startSession() },
                                modifier = Modifier.weight(1f).height(48.dp),
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Icon(imageVector = Icons.Default.PlayArrow, contentDescription = null)
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("Start Session")
                            }
                        }

                        SurveySessionStatus.ACTIVE -> {
                            OutlinedButton(
                                onClick = { sessionViewModel.pauseSession() },
                                modifier = Modifier.weight(1f).height(48.dp),
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Icon(imageVector = Icons.Default.Pause, contentDescription = null)
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("Pause")
                            }

                            Button(
                                onClick = { sessionViewModel.completeSession() },
                                modifier = Modifier.weight(1f).height(48.dp),
                                shape = RoundedCornerShape(10.dp),
                                colors = ButtonDefaults.buttonColors(containerColor = SignalRed)
                            ) {
                                Icon(imageVector = Icons.Default.Stop, contentDescription = null)
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("Complete")
                            }
                        }

                        SurveySessionStatus.PAUSED -> {
                            Button(
                                onClick = { sessionViewModel.resumeSession() },
                                modifier = Modifier.weight(1f).height(48.dp),
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Icon(imageVector = Icons.Default.PlayArrow, contentDescription = null)
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("Resume")
                            }

                            OutlinedButton(
                                onClick = { sessionViewModel.completeSession() },
                                modifier = Modifier.weight(1f).height(48.dp),
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Icon(imageVector = Icons.Default.CheckCircle, contentDescription = null)
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("Complete")
                            }
                        }

                        SurveySessionStatus.COMPLETED, SurveySessionStatus.CANCELLED -> {
                            Button(
                                onClick = { showCreateDialog = true },
                                modifier = Modifier.weight(1f).height(48.dp),
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Icon(imageVector = Icons.Default.Add, contentDescription = null)
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("New Session")
                            }

                            if (status == SurveySessionStatus.COMPLETED) {
                                OutlinedButton(
                                    onClick = { sessionViewModel.exportDatasetZip() },
                                    modifier = Modifier.weight(1f).height(48.dp),
                                    shape = RoundedCornerShape(10.dp)
                                ) {
                                    Icon(imageVector = Icons.Default.Archive, contentDescription = null)
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text("Export ZIP")
                                }
                            }
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // --- 4. Observation Telemetry Counters ---
        Text(
            text = "Multimodal Observation Accounting",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold
        )
        Spacer(modifier = Modifier.height(8.dp))

        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            TelemetryCountCard(
                title = "Wi-Fi Scans",
                count = currentSession?.wifiCount ?: 0,
                icon = Icons.Default.Wifi,
                modifier = Modifier.weight(1f)
            )
            TelemetryCountCard(
                title = "IMU Samples",
                count = currentSession?.sensorCount ?: 0,
                icon = Icons.Default.Sensors,
                modifier = Modifier.weight(1f)
            )
            TelemetryCountCard(
                title = "Photos",
                count = currentSession?.photoCount ?: 0,
                icon = Icons.Default.CameraAlt,
                modifier = Modifier.weight(1f)
            )
            TelemetryCountCard(
                title = "Annotations",
                count = currentSession?.annotationCount ?: 0,
                icon = Icons.Default.Edit,
                modifier = Modifier.weight(1f)
            )
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Actions: Annotations, Simulation & Export
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = { showAnnotationDialog = true },
                enabled = currentSession != null && (isActive || isPaused),
                modifier = Modifier.weight(1f),
                shape = RoundedCornerShape(10.dp)
            ) {
                Icon(imageVector = Icons.Default.AddCircle, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(modifier = Modifier.width(6.dp))
                Text("Add Annotation")
            }

            OutlinedButton(
                onClick = { sessionViewModel.addMockObservation() },
                enabled = isActive,
                modifier = Modifier.weight(1f),
                shape = RoundedCornerShape(10.dp)
            ) {
                Text("Simulate Packet")
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = { sessionViewModel.exportDatasetZip() },
                enabled = currentSession != null && !isExporting,
                modifier = Modifier.weight(1.4f),
                shape = RoundedCornerShape(10.dp),
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary)
            ) {
                if (isExporting) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), color = Color.White)
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Exporting...")
                } else {
                    Icon(imageVector = Icons.Default.Archive, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Validate & Export ZIP")
                }
            }

            if (currentSession != null) {
                OutlinedButton(
                    onClick = { showDeleteConfirmDialog = true },
                    modifier = Modifier.weight(0.8f),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Icon(imageVector = Icons.Default.Delete, contentDescription = null, tint = SignalRed, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Delete", color = SignalRed)
                }
            }
        }

        // Status Feedback Alert
        statusMessage?.let { msg ->
            Spacer(modifier = Modifier.height(10.dp))
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
            ) {
                Row(
                    modifier = Modifier.padding(10.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(imageVector = Icons.Default.Info, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(20.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(text = msg, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                    IconButton(onClick = { sessionViewModel.clearStatusMessage() }, modifier = Modifier.size(24.dp)) {
                        Icon(imageVector = Icons.Default.Close, contentDescription = "Dismiss", modifier = Modifier.size(16.dp))
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(18.dp))

        // --- 5. Chronological Observation Timeline ---
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Observation Timeline (${timeline.size})",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Icon(imageVector = Icons.Default.Timeline, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
        }

        Spacer(modifier = Modifier.height(8.dp))

        if (timeline.isEmpty()) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Icon(imageVector = Icons.Default.Timeline, contentDescription = null, tint = Color.Gray, modifier = Modifier.size(36.dp))
                    Spacer(modifier = Modifier.height(6.dp))
                    Text("No observations in session timeline", style = MaterialTheme.typography.bodyMedium)
                    Text("Start scanning or capturing photos to record stream events", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
                }
            }
        } else {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                // Show latest up to 50 items for UI responsiveness
                timeline.takeLast(50).reversed().forEach { item ->
                    TimelineItemCard(
                        item = item,
                        onClick = { inspectingItem = item }
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        // --- 6. Previous Sessions History ---
        if (allSessions.size > 1) {
            Text(
                text = "Session History (${allSessions.size})",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(8.dp))
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                allSessions.forEach { sess ->
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { sessionViewModel.startSession(sess.sessionId) },
                        colors = CardDefaults.cardColors(
                            containerColor = if (sess.sessionId == currentSession?.sessionId)
                                MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f)
                            else MaterialTheme.colorScheme.surface
                        )
                    ) {
                        Row(
                            modifier = Modifier.padding(10.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(sess.sessionName, fontWeight = FontWeight.SemiBold)
                                Text("${sess.sessionId} • ${sess.status.name} • ${sess.totalObservations} obs", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
                            }
                            StatusBadge(text = sess.status.displayName, color = MaterialTheme.colorScheme.primary)
                        }
                    }
                }
            }
            Spacer(modifier = Modifier.height(16.dp))
        }

        // --- 7. Server Telemetry Sync Card ---
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
        ) {
            Column(modifier = Modifier.padding(14.dp)) {
                Text("Backend Synchronization", fontWeight = FontWeight.Bold)
                Spacer(modifier = Modifier.height(6.dp))
                OutlinedTextField(
                    value = urlInput,
                    onValueChange = { urlInput = it },
                    label = { Text("Base URL") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true
                )
                Spacer(modifier = Modifier.height(8.dp))
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = { mainViewModel.updateBackendUrl(urlInput) }, modifier = Modifier.weight(1f)) {
                        Text("Apply")
                    }
                    Button(onClick = { sessionViewModel.syncWithBackend() }, enabled = currentSession != null, modifier = Modifier.weight(1f)) {
                        Icon(imageVector = Icons.Default.CloudSync, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Sync")
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))
    }

    // --- Dialogs ---

    // 1. Create New Session Dialog
    if (showCreateDialog) {
        var name by remember { mutableStateOf("Survey Session ${allSessions.size + 1}") }
        var building by remember { mutableStateOf("") }
        var floor by remember { mutableStateOf("") }
        var room by remember { mutableStateOf("") }
        var notes by remember { mutableStateOf("") }

        AlertDialog(
            onDismissRequest = { showCreateDialog = false },
            title = { Text("Configure Survey Session") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(value = name, onValueChange = { name = it }, label = { Text("Session Name") }, singleLine = true, modifier = Modifier.fillMaxWidth())
                    OutlinedTextField(value = building, onValueChange = { building = it }, label = { Text("Building") }, placeholder = { Text("e.g. TI-Engineering") }, singleLine = true, modifier = Modifier.fillMaxWidth())
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedTextField(value = floor, onValueChange = { floor = it }, label = { Text("Floor") }, placeholder = { Text("e.g. 2") }, singleLine = true, modifier = Modifier.weight(1f))
                        OutlinedTextField(value = room, onValueChange = { room = it }, label = { Text("Room / Area") }, placeholder = { Text("e.g. 204") }, singleLine = true, modifier = Modifier.weight(1f))
                    }
                    OutlinedTextField(value = notes, onValueChange = { notes = it }, label = { Text("Notes (Optional)") }, modifier = Modifier.fillMaxWidth())
                }
            },
            confirmButton = {
                Button(onClick = {
                    sessionViewModel.createSession(name, building, floor, room, notes)
                    showCreateDialog = false
                }) {
                    Text("Create")
                }
            },
            dismissButton = {
                TextButton(onClick = { showCreateDialog = false }) { Text("Cancel") }
            }
        )
    }

    // 2. Add Annotation Dialog
    if (showAnnotationDialog) {
        var selectedType by remember { mutableStateOf(AnnotationType.CORRIDOR) }
        var desc by remember { mutableStateOf("") }
        var bldg by remember { mutableStateOf(currentSession?.buildingId.orEmpty()) }
        var flr by remember { mutableStateOf(currentSession?.floorId.orEmpty()) }
        var rm by remember { mutableStateOf(currentSession?.roomId.orEmpty()) }
        var isMenuExpanded by remember { mutableStateOf(false) }

        AlertDialog(
            onDismissRequest = { showAnnotationDialog = false },
            title = { Text("Add Manual Ground-Truth Annotation") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    ExposedDropdownMenuBox(
                        expanded = isMenuExpanded,
                        onExpandedChange = { isMenuExpanded = !isMenuExpanded }
                    ) {
                        OutlinedTextField(
                            value = selectedType.displayName,
                            onValueChange = {},
                            readOnly = true,
                            label = { Text("Annotation Type") },
                            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = isMenuExpanded) },
                            modifier = Modifier.menuAnchor().fillMaxWidth()
                        )
                        ExposedDropdownMenu(
                            expanded = isMenuExpanded,
                            onDismissRequest = { isMenuExpanded = false }
                        ) {
                            AnnotationType.entries.forEach { type ->
                                DropdownMenuItem(
                                    text = { Text(type.displayName) },
                                    onClick = {
                                        selectedType = type
                                        isMenuExpanded = false
                                    }
                                )
                            }
                        }
                    }

                    OutlinedTextField(value = desc, onValueChange = { desc = it }, label = { Text("Description") }, placeholder = { Text("e.g. Turn at north stairwell") }, modifier = Modifier.fillMaxWidth())
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedTextField(value = bldg, onValueChange = { bldg = it }, label = { Text("Building") }, modifier = Modifier.weight(1f))
                        OutlinedTextField(value = flr, onValueChange = { flr = it }, label = { Text("Floor") }, modifier = Modifier.weight(1f))
                    }
                    OutlinedTextField(value = rm, onValueChange = { rm = it }, label = { Text("Room / Area") }, modifier = Modifier.fillMaxWidth())
                }
            },
            confirmButton = {
                Button(onClick = {
                    sessionViewModel.addAnnotation(
                        type = selectedType,
                        description = desc.ifBlank { selectedType.displayName },
                        building = bldg,
                        floor = flr,
                        room = rm
                    )
                    showAnnotationDialog = false
                }) {
                    Text("Record Annotation")
                }
            },
            dismissButton = {
                TextButton(onClick = { showAnnotationDialog = false }) { Text("Cancel") }
            }
        )
    }

    // 3. Inspect Timeline Observation Dialog
    inspectingItem?.let { item ->
        AlertDialog(
            onDismissRequest = { inspectingItem = null },
            title = { Text(item.title) },
            text = {
                Column(modifier = Modifier.verticalScroll(rememberScrollState())) {
                    Text("Type: ${item.type.displayName}", fontWeight = FontWeight.Bold)
                    Text("Time: ${item.formattedTime} (${item.timeSourceDisplay})", style = MaterialTheme.typography.bodySmall)
                    Text("Uncertainty: ${item.uncertaintyDisplay}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.primary)
                    Spacer(modifier = Modifier.height(8.dp))
                    item.detailKeyValues.forEach { (k, v) ->
                        Row(modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp)) {
                            Text("$k: ", fontWeight = FontWeight.SemiBold, style = MaterialTheme.typography.bodySmall)
                            Text(v, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = { inspectingItem = null }) { Text("Close") }
            }
        )
    }

    // 4. Exported ZIP Success Dialog
    exportedZip?.let { zip ->
        AlertDialog(
            onDismissRequest = { sessionViewModel.clearExportedZip() },
            title = { Text("Dataset ZIP Exported") },
            text = {
                Column {
                    Text("File: ${zip.name}", fontWeight = FontWeight.Bold)
                    Text("Path: ${zip.absolutePath}", style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
                    Text("Size: ${zip.length() / 1024} KB", style = MaterialTheme.typography.bodySmall)
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("Contains validated manifest.json with SHA-256 hashes, CSV observations, and sanitized photos.", style = MaterialTheme.typography.bodySmall)
                }
            },
            confirmButton = {
                Button(onClick = {
                    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                    clipboard.setPrimaryClip(ClipData.newPlainText("ZIP Path", zip.absolutePath))
                    Toast.makeText(context, "Path copied", Toast.LENGTH_SHORT).show()
                    sessionViewModel.clearExportedZip()
                }) {
                    Text("Copy Path")
                }
            },
            dismissButton = {
                TextButton(onClick = { sessionViewModel.clearExportedZip() }) { Text("Done") }
            }
        )
    }

    // 5. Delete Session Confirmation Dialog
    if (showDeleteConfirmDialog && currentSession != null) {
        AlertDialog(
            onDismissRequest = { showDeleteConfirmDialog = false },
            title = { Text("Delete Survey Session?") },
            text = {
                Text("Are you sure you want to delete '${currentSession?.sessionName}' (${currentSession?.sessionId})? All associated observation journals and local images will be deleted from disk.")
            },
            confirmButton = {
                Button(
                    onClick = {
                        sessionViewModel.deleteSession(currentSession!!.sessionId)
                        showDeleteConfirmDialog = false
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = SignalRed)
                ) {
                    Text("Delete")
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteConfirmDialog = false }) { Text("Cancel") }
            }
        )
    }
}

@Composable
private fun TelemetryCountCard(
    title: String,
    count: Int,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(10.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier.padding(8.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(imageVector = icon, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(20.dp))
            Spacer(modifier = Modifier.height(2.dp))
            Text(text = count.toString(), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Text(text = title, style = MaterialTheme.typography.labelSmall, color = Color.Gray, fontSize = 10.sp)
        }
    }
}

@Composable
private fun TimelineItemCard(
    item: SessionTimelineItem,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth().clickable(onClick = onClick),
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Row(
            modifier = Modifier.padding(10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            val icon = when (item.type) {
                TimelineObservationType.WIFI_SCAN -> Icons.Default.Wifi
                TimelineObservationType.ACCELEROMETER,
                TimelineObservationType.GYROSCOPE,
                TimelineObservationType.MAGNETOMETER,
                TimelineObservationType.ORIENTATION -> Icons.Default.Sensors
                TimelineObservationType.PHOTO_CAPTURE -> Icons.Default.CameraAlt
                TimelineObservationType.ANNOTATION -> Icons.Default.Edit
            }

            Icon(imageVector = icon, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(22.dp))
            Spacer(modifier = Modifier.width(10.dp))

            Column(modifier = Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(item.title, fontWeight = FontWeight.SemiBold, style = MaterialTheme.typography.bodyMedium)
                    Spacer(modifier = Modifier.width(6.dp))
                    if (item.isMock) {
                        StatusBadge(text = "MOCK", color = SignalYellow)
                    }
                }
                Text(item.shortDescription, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text("${item.formattedTime} • ${item.uncertaintyDisplay}", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
            }
        }
    }
}
