package ai.indoor.app.ui.screens

import android.Manifest
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.BookmarkAdd
import androidx.compose.material.icons.filled.Code
import androidx.compose.material.icons.filled.FileDownload
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Security
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material.icons.filled.Wifi
import androidx.compose.material.icons.filled.WifiOff
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
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
import ai.indoor.app.domain.models.LocalizationResult
import ai.indoor.app.domain.models.WifiFingerprint
import ai.indoor.app.ui.components.AccessPointItem
import ai.indoor.app.ui.components.MockDataBadge
import ai.indoor.app.ui.components.StatusBadge
import ai.indoor.app.viewmodel.WifiScanViewModel
import ai.indoor.app.wifi.WifiScanState
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun WifiScanScreen(
    viewModel: WifiScanViewModel,
    modifier: Modifier = Modifier
) {
    val scanState by viewModel.scanState.collectAsState()
    val isMockMode by viewModel.isMockMode.collectAsState()
    val lastScanTime by viewModel.lastScanTime.collectAsState()
    val currentFingerprint by viewModel.currentFingerprint.collectAsState()
    val capturedFingerprints by viewModel.capturedFingerprints.collectAsState()
    val modelAdaptationNotice by viewModel.modelAdaptationNotice.collectAsState()
    val localizationResult by viewModel.localizationResult.collectAsState()
    val isLocalizing by viewModel.isLocalizing.collectAsState()

    val exportPreview by viewModel.exportPreview.collectAsState()
    val exportFormat by viewModel.exportFormat.collectAsState()

    val manualBuilding by viewModel.manualBuilding.collectAsState()
    val manualFloor by viewModel.manualFloor.collectAsState()
    val manualRoom by viewModel.manualRoom.collectAsState()
    val manualNotes by viewModel.manualNotes.collectAsState()

    var showLabelingSection by remember { mutableStateOf(false) }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            viewModel.triggerScan()
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        // Header
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Wi-Fi Fingerprint Collector",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "802.11 beacon observation & calibration dataset collection",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            if (isMockMode) {
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

        // Scan Action Card
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
                            text = "Detected Beacons: ${currentFingerprint?.count ?: 0}",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        val freshInfo = currentFingerprint?.let {
                            "(${it.freshCount} fresh, ${it.cachedCount} cached)"
                        } ?: ""
                        val timeStr = lastScanTime?.let {
                            SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date(it))
                        } ?: "Never"
                        Text(
                            text = "Last scan: $timeStr $freshInfo",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }

                    Row(verticalAlignment = Alignment.CenterVertically) {
                        IconButton(
                            onClick = { viewModel.triggerScan() },
                            enabled = scanState !is WifiScanState.Scanning
                        ) {
                            Icon(imageVector = Icons.Default.Refresh, contentDescription = "Refresh Scan")
                        }

                        Button(
                            onClick = { viewModel.triggerScan() },
                            enabled = scanState !is WifiScanState.Scanning,
                            shape = RoundedCornerShape(8.dp)
                        ) {
                            if (scanState is WifiScanState.Scanning) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(18.dp),
                                    color = MaterialTheme.colorScheme.onPrimary,
                                    strokeWidth = 2.dp
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("Scanning")
                            } else {
                                Icon(imageVector = Icons.Default.Wifi, contentDescription = null)
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("Start Scan")
                            }
                        }
                    }
                }

                // Status banner row
                Spacer(modifier = Modifier.height(8.dp))
                when (val s = scanState) {
                    is WifiScanState.Idle -> {
                        Text(
                            text = "Status: Radio idle. Press 'Start Scan' to trigger an 802.11 beacon sweep.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    is WifiScanState.Scanning -> {
                        Text(
                            text = "Status: Performing active 802.11 radio sweep via WifiManager...",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                    is WifiScanState.Success -> {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = Icons.Default.Info,
                                contentDescription = null,
                                tint = if (s.isFresh) Color(0xFF16A34A) else Color(0xFFD97706),
                                modifier = Modifier.size(16.dp)
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = if (s.isFresh)
                                    "Fresh Radio Sweep: ${s.fingerprint.count} access points received from OS."
                                else
                                    "Cached Results: Android OS served cached sweep data.",
                                style = MaterialTheme.typography.bodySmall,
                                color = if (s.isFresh) Color(0xFF16A34A) else Color(0xFFD97706)
                            )
                        }
                    }
                    is WifiScanState.Throttled -> {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(imageVector = Icons.Default.Warning, contentDescription = null, tint = Color(0xFFD97706), modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = s.message,
                                style = MaterialTheme.typography.bodySmall,
                                color = Color(0xFFB45309)
                            )
                        }
                    }
                    is WifiScanState.PermissionRequired -> {
                        Column {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(imageVector = Icons.Default.Security, contentDescription = null, tint = MaterialTheme.colorScheme.error, modifier = Modifier.size(16.dp))
                                Spacer(modifier = Modifier.width(4.dp))
                                Text(
                                    text = "ACCESS_FINE_LOCATION permission required by Android to read Wi-Fi beacons.",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.error
                                )
                            }
                            Spacer(modifier = Modifier.height(4.dp))
                            OutlinedButton(
                                onClick = { permissionLauncher.launch(Manifest.permission.ACCESS_FINE_LOCATION) },
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text("Grant Location Permission")
                            }
                        }
                    }
                    is WifiScanState.LocationServicesRequired -> {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(imageVector = Icons.Default.LocationOn, contentDescription = null, tint = MaterialTheme.colorScheme.error, modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = s.message,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.error
                            )
                        }
                    }
                    is WifiScanState.WifiDisabled -> {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(imageVector = Icons.Default.WifiOff, contentDescription = null, tint = MaterialTheme.colorScheme.error, modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = s.message,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.error
                            )
                        }
                    }
                    is WifiScanState.Error -> {
                        Text(
                            text = "Error: ${s.message}",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.error
                        )
                    }
                }
            }
        }

        // Section 8: UJIIndoorLoc Compatibility Notice Card
        modelAdaptationNotice?.let { notice ->
            Spacer(modifier = Modifier.height(10.dp))
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(10.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.45f)
                )
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.Warning,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.tertiary,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "Model Calibration Notice",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.tertiary
                        )
                    }
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = notice,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onTertiaryContainer
                    )
                }
            }
        }

        // Demo Localization Button (Only available if in mock synthetic mode)
        if (isMockMode && currentFingerprint != null && currentFingerprint!!.accessPoints.isNotEmpty()) {
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedButton(
                onClick = { viewModel.requestDemoLocalization() },
                enabled = !isLocalizing,
                modifier = Modifier.fillMaxWidth()
            ) {
                if (isLocalizing) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Computing Demo Localization...")
                } else {
                    Icon(imageVector = Icons.Default.LocationOn, contentDescription = null)
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Run Demo Localization (Synthetic Campus APs)")
                }
            }
        }

        // Localization Result Card (if demo requested)
        localizationResult?.let { res ->
            Spacer(modifier = Modifier.height(10.dp))
            DemoLocalizationCard(result = res)
        }

        Spacer(modifier = Modifier.height(10.dp))

        // Ground-Truth Labeling & Export Controls Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.BookmarkAdd,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = "Ground-Truth Capture & Export",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.weight(1f))
                    TextButton(onClick = { showLabelingSection = !showLabelingSection }) {
                        Text(if (showLabelingSection) "Hide Labels" else "Edit Labels")
                    }
                }

                if (showLabelingSection) {
                    Spacer(modifier = Modifier.height(6.dp))
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedTextField(
                            value = manualBuilding,
                            onValueChange = { viewModel.updateManualLabels(it, manualFloor, manualRoom, manualNotes) },
                            label = { Text("Building") },
                            modifier = Modifier.weight(1f),
                            singleLine = true
                        )
                        OutlinedTextField(
                            value = manualFloor,
                            onValueChange = { viewModel.updateManualLabels(manualBuilding, it, manualRoom, manualNotes) },
                            label = { Text("Floor") },
                            modifier = Modifier.weight(1f),
                            singleLine = true
                        )
                    }
                    Spacer(modifier = Modifier.height(6.dp))
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedTextField(
                            value = manualRoom,
                            onValueChange = { viewModel.updateManualLabels(manualBuilding, manualFloor, it, manualNotes) },
                            label = { Text("Room / Landmark") },
                            modifier = Modifier.weight(1f),
                            singleLine = true
                        )
                        OutlinedTextField(
                            value = manualNotes,
                            onValueChange = { viewModel.updateManualLabels(manualBuilding, manualFloor, manualRoom, it) },
                            label = { Text("Notes") },
                            modifier = Modifier.weight(1.2f),
                            singleLine = true
                        )
                    }
                }

                Spacer(modifier = Modifier.height(10.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Button(
                        onClick = { viewModel.saveCurrentFingerprint() },
                        enabled = currentFingerprint != null && currentFingerprint!!.accessPoints.isNotEmpty(),
                        modifier = Modifier.weight(1.3f),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Icon(imageVector = Icons.Default.BookmarkAdd, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Save (${capturedFingerprints.size})")
                    }

                    OutlinedButton(
                        onClick = { viewModel.exportAsJson() },
                        enabled = currentFingerprint != null || capturedFingerprints.isNotEmpty(),
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("JSON")
                    }

                    OutlinedButton(
                        onClick = { viewModel.exportAsCsv() },
                        enabled = currentFingerprint != null || capturedFingerprints.isNotEmpty(),
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("CSV")
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Observed Access Points Title
        Text(
            text = "Observed Access Points (${currentFingerprint?.count ?: 0})",
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.Bold
        )

        Spacer(modifier = Modifier.height(6.dp))

        // AP List or Empty State
        val aps = currentFingerprint?.accessPoints.orEmpty()
        if (aps.isEmpty()) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Icon(
                        imageVector = Icons.Default.Wifi,
                        contentDescription = null,
                        tint = Color.LightGray,
                        modifier = Modifier.size(54.dp)
                    )
                    Spacer(modifier = Modifier.height(10.dp))
                    Text(
                        text = "No Wi-Fi fingerprints captured yet.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Text(
                        text = "Press 'Start Scan' above to observe 802.11 beacon transmissions.",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color.Gray
                    )
                }
            }
        } else {
            LazyColumn(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                contentPadding = PaddingValues(bottom = 16.dp)
            ) {
                items(
                    items = aps,
                    key = { it.bssid + "_" + it.timestamp }
                ) { ap ->
                    AccessPointItem(accessPoint = ap)
                }
            }
        }
    }

    // Export Preview Modal Dialog
    if (exportPreview != null) {
        AlertDialog(
            onDismissRequest = { viewModel.clearExportPreview() },
            title = {
                Text("Exported Fingerprints ($exportFormat)")
            },
            text = {
                Column {
                    Text(
                        text = "Ready for dataset export, calibration mapping, or offline analysis:",
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

@Composable
private fun DemoLocalizationCard(
    result: LocalizationResult,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.5f)
        )
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Default.LocationOn,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = "Demo Estimate (${result.modelType})",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary
                )
                Spacer(modifier = Modifier.weight(1f))
                StatusBadge(text = "Confidence: ${(result.confidence * 100).toInt()}%")
            }

            Spacer(modifier = Modifier.height(8.dp))

            Row(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(text = "Building ID", style = MaterialTheme.typography.labelSmall)
                    Text(text = "Building ${result.buildingId}", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                }
                Column(modifier = Modifier.weight(1f)) {
                    Text(text = "Floor", style = MaterialTheme.typography.labelSmall)
                    Text(text = "Floor ${result.floor}", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                }
                Column(modifier = Modifier.weight(1.2f)) {
                    Text(text = "Native Coordinates", style = MaterialTheme.typography.labelSmall)
                    Text(
                        text = "X: %.1f\nY: %.1f".format(result.longitude, result.latitude),
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }
        }
    }
}
