package ai.indoor.app.ui.screens

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Camera
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Photo
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Security
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.Warning
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
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import ai.indoor.app.domain.models.CapturedPhotoMetadata
import ai.indoor.app.domain.models.FaceDetectionStatus
import ai.indoor.app.domain.models.LandmarkType
import ai.indoor.app.ui.components.StatusBadge
import ai.indoor.app.ui.theme.SignalGreen
import ai.indoor.app.ui.theme.SignalRed
import ai.indoor.app.ui.theme.SignalYellow
import ai.indoor.app.viewmodel.CaptureFlowState
import ai.indoor.app.viewmodel.ImageCaptureViewModel
import ai.indoor.app.viewmodel.PreviewState
import ai.indoor.app.viewmodel.SessionViewModel
import java.io.File

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ImageCaptureScreen(
    viewModel: ImageCaptureViewModel,
    sessionViewModel: SessionViewModel? = null,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scrollState = rememberScrollState()

    val previewState by viewModel.previewState.collectAsState()
    val captureFlowState by viewModel.captureFlowState.collectAsState()
    val capturedPhotos by viewModel.capturedPhotos.collectAsState()
    val currentDraft by viewModel.currentDraft.collectAsState()
    val faceBlurEnabled by viewModel.faceBlurEnabled.collectAsState()
    val statusMessage by viewModel.statusMessage.collectAsState()

    val currentSession by (sessionViewModel?.currentSession?.collectAsState() ?: remember { mutableStateOf(null) })
    val activeSessionId = currentSession?.sessionId

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        )
    }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasCameraPermission = isGranted
        if (isGranted) {
            Toast.makeText(context, "Camera permission granted", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(context, "Camera permission is required to capture photos", Toast.LENGTH_LONG).show()
        }
    }

    // Stop camera preview when user navigates away from this screen
    DisposableEffect(lifecycleOwner) {
        onDispose {
            viewModel.stopCameraPreview()
        }
    }

    // Export dialog state
    var exportDialogContent by remember { mutableStateOf<Pair<String, String>?>(null) } // Title to Content

    // Delete photo confirmation state
    var photoToDelete by remember { mutableStateOf<CapturedPhotoMetadata?>(null) }

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(16.dp)
    ) {
        // --- 1. Screen Header ---
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Landmark Photography",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "User-initiated, privacy-sanitized visual landmark collection",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            if (currentSession?.isRunning == true) {
                StatusBadge(
                    text = "Session ${currentSession?.sessionId?.takeLast(8)}",
                    color = SignalGreen
                )
            } else {
                StatusBadge(
                    text = "Standalone Capture",
                    color = MaterialTheme.colorScheme.outline
                )
            }
        }

        Spacer(modifier = Modifier.height(14.dp))

        // --- 2. Privacy Guarantee Banner ---
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.45f)
            )
        ) {
            Row(
                modifier = Modifier.padding(12.dp),
                verticalAlignment = Alignment.Top
            ) {
                Icon(
                    imageVector = Icons.Default.Security,
                    contentDescription = "Privacy Shield",
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(24.dp).padding(top = 2.dp)
                )
                Spacer(modifier = Modifier.width(10.dp))
                Column {
                    Text(
                        text = "Zero-Background Privacy Guarantee",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.primary
                    )
                    Text(
                        text = "• Camera preview runs on-demand only (never starts automatically)\n" +
                               "• EXIF GPS coordinates and device serials are permanently stripped\n" +
                               "• On-device face detection applies local privacy blur to bystanders\n" +
                               "• Photos are stored exclusively in app-private storage (no cloud upload)",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // --- 3. Status Toast / Feedback Message ---
        statusMessage?.let { msg ->
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 12.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            ) {
                Row(
                    modifier = Modifier.padding(10.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.Info,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(text = msg, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                    IconButton(
                        onClick = { viewModel.clearStatusMessage() },
                        modifier = Modifier.size(24.dp)
                    ) {
                        Icon(imageVector = Icons.Default.Close, contentDescription = "Dismiss", modifier = Modifier.size(16.dp))
                    }
                }
            }
        }

        // --- 4. Camera Viewfinder / Preview Section ---
        Text(
            text = "Camera Viewfinder",
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.Bold
        )
        Spacer(modifier = Modifier.height(6.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(260.dp)
                .clip(RoundedCornerShape(16.dp))
                .background(Color.Black)
                .border(
                    width = 1.dp,
                    color = MaterialTheme.colorScheme.outlineVariant,
                    shape = RoundedCornerShape(16.dp)
                ),
            contentAlignment = Alignment.Center
        ) {
            when (previewState) {
                PreviewState.INACTIVE -> {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.PhotoCamera,
                            contentDescription = null,
                            tint = Color.LightGray,
                            modifier = Modifier.size(52.dp)
                        )
                        Spacer(modifier = Modifier.height(10.dp))
                        Text(
                            text = "Camera is currently inactive",
                            style = MaterialTheme.typography.titleSmall,
                            color = Color.White,
                            fontWeight = FontWeight.SemiBold
                        )
                        Text(
                            text = "Tap 'Start Preview' below to initialize the camera viewfinder",
                            style = MaterialTheme.typography.bodySmall,
                            color = Color.Gray,
                            textAlign = TextAlign.Center
                        )
                    }
                }

                PreviewState.STARTING -> {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator(color = Color.White)
                        Spacer(modifier = Modifier.height(10.dp))
                        Text("Starting camera...", color = Color.White, style = MaterialTheme.typography.bodyMedium)
                    }
                }

                PreviewState.ACTIVE -> {
                    // CameraX PreviewView hosted inside AndroidView
                    AndroidView(
                        factory = { ctx ->
                            PreviewView(ctx).apply {
                                scaleType = PreviewView.ScaleType.FILL_CENTER
                            }
                        },
                        modifier = Modifier.fillMaxSize(),
                        update = { previewView ->
                            viewModel.startCameraPreview(lifecycleOwner, previewView, hasCameraPermission)
                        }
                    )

                    // Target Crosshair / Grid Overlay to assist indoor alignment
                    CanvasOverlayCrosshair()

                    // Floating stop preview button
                    IconButton(
                        onClick = { viewModel.stopCameraPreview() },
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .padding(8.dp)
                            .background(Color.Black.copy(alpha = 0.5f), CircleShape)
                    ) {
                        Icon(imageVector = Icons.Default.Close, contentDescription = "Stop Preview", tint = Color.White)
                    }
                }

                PreviewState.PERMISSION_REQUIRED -> {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.padding(20.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Lock,
                            contentDescription = null,
                            tint = SignalYellow,
                            modifier = Modifier.size(44.dp)
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "Camera Permission Required",
                            style = MaterialTheme.typography.titleSmall,
                            color = Color.White,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "Camera access is needed exclusively for manual indoor landmark photography.",
                            style = MaterialTheme.typography.bodySmall,
                            color = Color.LightGray,
                            textAlign = TextAlign.Center
                        )
                        Spacer(modifier = Modifier.height(12.dp))
                        Button(
                            onClick = { cameraPermissionLauncher.launch(Manifest.permission.CAMERA) },
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                        ) {
                            Text("Grant Camera Permission")
                        }
                    }
                }

                PreviewState.UNAVAILABLE -> {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Warning,
                            contentDescription = null,
                            tint = SignalYellow,
                            modifier = Modifier.size(44.dp)
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "No Camera Hardware Detected",
                            style = MaterialTheme.typography.titleSmall,
                            color = Color.White,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "Using synthetic landmark generator for benchmarking and testing.",
                            style = MaterialTheme.typography.bodySmall,
                            color = Color.LightGray,
                            textAlign = TextAlign.Center
                        )
                    }
                }

                PreviewState.ERROR -> {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Warning,
                            contentDescription = null,
                            tint = SignalRed,
                            modifier = Modifier.size(44.dp)
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "Camera Hardware Unavailable",
                            style = MaterialTheme.typography.titleSmall,
                            color = Color.White,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(10.dp))
                        Button(onClick = { viewModel.stopCameraPreview() }) {
                            Text("Reset Viewfinder")
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // --- 5. Viewfinder Action Controls ---
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            if (previewState == PreviewState.ACTIVE) {
                Button(
                    onClick = { viewModel.takePhoto() },
                    modifier = Modifier
                        .weight(1f)
                        .height(50.dp),
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                ) {
                    Icon(imageVector = Icons.Default.CameraAlt, contentDescription = "Shutter")
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Capture Landmark Frame", fontWeight = FontWeight.Bold)
                }

                OutlinedButton(
                    onClick = { viewModel.stopCameraPreview() },
                    modifier = Modifier.height(50.dp),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Icon(imageVector = Icons.Default.Stop, contentDescription = "Stop")
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Stop")
                }
            } else {
                Button(
                    onClick = {
                        if (hasCameraPermission) {
                            // If previewView is not yet mounted, we can request active state
                            viewModel.startCameraPreview(
                                lifecycleOwner = lifecycleOwner,
                                previewView = PreviewView(context),
                                hasPermission = true
                            )
                        } else {
                            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                        }
                    },
                    modifier = Modifier
                        .weight(1f)
                        .height(50.dp),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Icon(imageVector = Icons.Default.CameraAlt, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(if (hasCameraPermission) "Start Camera Preview" else "Grant & Start Preview")
                }

                OutlinedButton(
                    onClick = { viewModel.takePhoto() },
                    modifier = Modifier.height(50.dp),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Icon(imageVector = Icons.Default.Photo, contentDescription = null)
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Test Mock Capture")
                }
            }
        }

        // --- 6. Sanitizing State Spinner ---
        if (captureFlowState is CaptureFlowState.Sanitizing) {
            Spacer(modifier = Modifier.height(16.dp))
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
            ) {
                Row(
                    modifier = Modifier.padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(24.dp))
                    Spacer(modifier = Modifier.width(14.dp))
                    Column {
                        Text("Sanitizing Photograph...", fontWeight = FontWeight.SemiBold)
                        Text(
                            "Stripping EXIF GPS tags & applying on-device privacy blur",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        // --- 7. Saved Photos Catalog ---
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Captured Survey Photos (${capturedPhotos.size})",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )

            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                TextButton(
                    onClick = {
                        val json = viewModel.exportJson(activeSessionId)
                        exportDialogContent = "Survey Photos Metadata (JSON)" to json
                    },
                    enabled = capturedPhotos.isNotEmpty()
                ) {
                    Icon(imageVector = Icons.Default.Download, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("JSON")
                }

                TextButton(
                    onClick = {
                        val csv = viewModel.exportCsv(activeSessionId)
                        exportDialogContent = "Survey Photos Index (CSV)" to csv
                    },
                    enabled = capturedPhotos.isNotEmpty()
                ) {
                    Icon(imageVector = Icons.Default.Download, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("CSV")
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        if (capturedPhotos.isEmpty()) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Icon(
                        imageVector = Icons.Default.Photo,
                        contentDescription = null,
                        tint = Color.Gray,
                        modifier = Modifier.size(40.dp)
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "No landmark photos recorded yet",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Medium
                    )
                    Text(
                        text = "Capture door signs, corridor junctions, and room landmarks during your survey",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center
                    )
                }
            }
        } else {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                capturedPhotos.forEach { photo ->
                    CapturedPhotoCard(
                        photo = photo,
                        onDeleteClick = { photoToDelete = photo }
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))
    }

    // --- 8. Review and Landmark Annotation Modal Dialog ---
    val reviewingState = captureFlowState as? CaptureFlowState.Reviewing
    if (reviewingState != null) {
        ReviewAndAnnotateDialog(
            reviewState = reviewingState,
            currentDraft = currentDraft,
            onDraftChange = { building, floor, room, landmarkType, desc, notes ->
                viewModel.updateDraft(building, floor, room, landmarkType, desc, notes)
            },
            onToggleFaceBlur = { enabled -> viewModel.toggleFaceBlur(enabled) },
            onRetake = { viewModel.retakePhoto() },
            onConfirmAndSave = { viewModel.confirmAndSavePhoto(activeSessionId) }
        )
    }

    // --- 9. Export Dialog ---
    exportDialogContent?.let { (title, content) ->
        AlertDialog(
            onDismissRequest = { exportDialogContent = null },
            title = { Text(title) },
            text = {
                Column(modifier = Modifier.verticalScroll(rememberScrollState())) {
                    Text(
                        text = content,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 11.sp,
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(8.dp))
                            .padding(10.dp)
                    )
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                        clipboard.setPrimaryClip(ClipData.newPlainText(title, content))
                        Toast.makeText(context, "Copied to clipboard", Toast.LENGTH_SHORT).show()
                        exportDialogContent = null
                    }
                ) {
                    Icon(imageVector = Icons.Default.ContentCopy, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Copy to Clipboard")
                }
            },
            dismissButton = {
                TextButton(onClick = { exportDialogContent = null }) {
                    Text("Close")
                }
            }
        )
    }

    // --- 10. Delete Confirmation Dialog ---
    photoToDelete?.let { photo ->
        AlertDialog(
            onDismissRequest = { photoToDelete = null },
            title = { Text("Delete Landmark Photo?") },
            text = {
                Text("Are you sure you want to delete '${photo.filename}' (${photo.displayLocation}) from local storage? This action cannot be undone.")
            },
            confirmButton = {
                Button(
                    onClick = {
                        viewModel.deletePhoto(photo.photoId)
                        photoToDelete = null
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = SignalRed)
                ) {
                    Text("Delete")
                }
            },
            dismissButton = {
                TextButton(onClick = { photoToDelete = null }) {
                    Text("Cancel")
                }
            }
        )
    }
}

/**
 * Overlay crosshair for indoor alignment.
 */
@Composable
private fun CanvasOverlayCrosshair() {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        // Horizontal guideline
        Box(
            modifier = Modifier
                .width(80.dp)
                .height(1.dp)
                .background(Color.White.copy(alpha = 0.5f))
        )
        // Vertical guideline
        Box(
            modifier = Modifier
                .width(1.dp)
                .height(80.dp)
                .background(Color.White.copy(alpha = 0.5f))
        )
    }
}

/**
 * Review, landmark annotation, and privacy consent dialog.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ReviewAndAnnotateDialog(
    reviewState: CaptureFlowState.Reviewing,
    currentDraft: ai.indoor.app.viewmodel.LandmarkMetadataDraft,
    onDraftChange: (String, String, String, LandmarkType, String, String) -> Unit,
    onToggleFaceBlur: (Boolean) -> Unit,
    onRetake: () -> Unit,
    onConfirmAndSave: () -> Unit
) {
    var building by remember { mutableStateOf(currentDraft.building) }
    var floor by remember { mutableStateOf(currentDraft.floor) }
    var room by remember { mutableStateOf(currentDraft.room) }
    var selectedLandmarkType by remember { mutableStateOf(currentDraft.landmarkType) }
    var description by remember { mutableStateOf(currentDraft.description) }
    var notes by remember { mutableStateOf(currentDraft.notes) }

    var isDropdownExpanded by remember { mutableStateOf(false) }

    AlertDialog(
        onDismissRequest = onRetake,
        title = {
            Text("Review Landmark Photograph", fontWeight = FontWeight.Bold)
        },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
            ) {
                // Image Preview Box
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(180.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(Color.Black),
                    contentAlignment = Alignment.Center
                ) {
                    if (reviewState.previewBitmap != null) {
                        Image(
                            bitmap = reviewState.previewBitmap.asImageBitmap(),
                            contentDescription = "Captured Frame Preview",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Fit
                        )
                    } else {
                        Icon(imageVector = Icons.Default.Photo, contentDescription = null, tint = Color.Gray, modifier = Modifier.size(48.dp))
                    }
                }

                Spacer(modifier = Modifier.height(10.dp))

                // On-Device Privacy Status Card
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Column(modifier = Modifier.padding(10.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = Icons.Default.Security,
                                contentDescription = null,
                                tint = SignalGreen,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = "Privacy & Sanitization Status",
                                style = MaterialTheme.typography.labelLarge,
                                fontWeight = FontWeight.Bold
                            )
                        }
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "✓ EXIF GPS coordinates will be stripped\n" +
                                   "✓ Hardware serial & device identifiers will be removed",
                            style = MaterialTheme.typography.bodySmall
                        )

                        Spacer(modifier = Modifier.height(6.dp))

                        // Face detection status
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = "On-Device Face Blur",
                                    style = MaterialTheme.typography.bodySmall,
                                    fontWeight = FontWeight.SemiBold
                                )
                                Text(
                                    text = when (reviewState.faceStatus) {
                                        FaceDetectionStatus.SUCCEEDED -> "${reviewState.detectedFacesCount} face(s) detected in scene"
                                        FaceDetectionStatus.NO_FACES_DETECTED -> "0 faces detected in scene"
                                        FaceDetectionStatus.UNAVAILABLE -> "Face detector unavailable"
                                        FaceDetectionStatus.FAILED -> "Detection error"
                                        FaceDetectionStatus.NOT_ATTEMPTED -> "Not evaluated"
                                    },
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }

                            Switch(
                                checked = reviewState.isBlurEnabled,
                                onCheckedChange = { onToggleFaceBlur(it) }
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(14.dp))

                // Landmark Annotation Fields
                Text(
                    text = "Landmark Ground-Truth Labels",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold
                )

                Spacer(modifier = Modifier.height(8.dp))

                // Landmark Type Dropdown
                ExposedDropdownMenuBox(
                    expanded = isDropdownExpanded,
                    onExpandedChange = { isDropdownExpanded = !isDropdownExpanded }
                ) {
                    OutlinedTextField(
                        value = selectedLandmarkType.displayName,
                        onValueChange = {},
                        readOnly = true,
                        label = { Text("Landmark Type") },
                        trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = isDropdownExpanded) },
                        modifier = Modifier
                            .menuAnchor()
                            .fillMaxWidth()
                    )

                    ExposedDropdownMenu(
                        expanded = isDropdownExpanded,
                        onDismissRequest = { isDropdownExpanded = false }
                    ) {
                        LandmarkType.entries.forEach { type ->
                            DropdownMenuItem(
                                text = { Text(type.displayName) },
                                onClick = {
                                    selectedLandmarkType = type
                                    isDropdownExpanded = false
                                    onDraftChange(building, floor, room, type, description, notes)
                                }
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    OutlinedTextField(
                        value = building,
                        onValueChange = {
                            building = it
                            onDraftChange(it, floor, room, selectedLandmarkType, description, notes)
                        },
                        label = { Text("Building") },
                        placeholder = { Text("e.g. TI-Engineering") },
                        modifier = Modifier.weight(1f)
                    )

                    OutlinedTextField(
                        value = floor,
                        onValueChange = {
                            floor = it
                            onDraftChange(building, it, room, selectedLandmarkType, description, notes)
                        },
                        label = { Text("Floor") },
                        placeholder = { Text("e.g. 2") },
                        modifier = Modifier.weight(1f)
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                OutlinedTextField(
                    value = room,
                    onValueChange = {
                        room = it
                        onDraftChange(building, floor, it, selectedLandmarkType, description, notes)
                    },
                    label = { Text("Room / Area") },
                    placeholder = { Text("e.g. Room 204 or North Corridor") },
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(modifier = Modifier.height(8.dp))

                OutlinedTextField(
                    value = description,
                    onValueChange = {
                        description = it
                        onDraftChange(building, floor, room, selectedLandmarkType, it, notes)
                    },
                    label = { Text("Landmark Description") },
                    placeholder = { Text("e.g. Room doorway with overhead plaque") },
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(modifier = Modifier.height(8.dp))

                OutlinedTextField(
                    value = notes,
                    onValueChange = {
                        notes = it
                        onDraftChange(building, floor, room, selectedLandmarkType, description, it)
                    },
                    label = { Text("Surveyor Notes (Optional)") },
                    placeholder = { Text("e.g. Captured facing East, artificial lighting") },
                    modifier = Modifier.fillMaxWidth()
                )
            }
        },
        confirmButton = {
            Button(
                onClick = onConfirmAndSave,
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
            ) {
                Icon(imageVector = Icons.Default.Check, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(modifier = Modifier.width(6.dp))
                Text("Confirm & Save")
            }
        },
        dismissButton = {
            OutlinedButton(onClick = onRetake) {
                Icon(imageVector = Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(modifier = Modifier.width(6.dp))
                Text("Retake")
            }
        }
    )
}

/**
 * Card displaying an individual captured photograph.
 */
@Composable
private fun CapturedPhotoCard(
    photo: CapturedPhotoMetadata,
    onDeleteClick: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Thumbnail / Icon
            Box(
                modifier = Modifier
                    .size(60.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.5f)),
                contentAlignment = Alignment.Center
            ) {
                val file = File(photo.filePath)
                var bitmap by remember { mutableStateOf<Bitmap?>(null) }
                LaunchedEffect(photo.filePath) {
                    if (file.exists()) {
                        bitmap = BitmapFactory.decodeFile(file.absolutePath)
                    }
                }

                if (bitmap != null) {
                    Image(
                        bitmap = bitmap!!.asImageBitmap(),
                        contentDescription = null,
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop
                    )
                } else {
                    Icon(
                        imageVector = Icons.Default.Photo,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(28.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.width(12.dp))

            // Information Column
            Column(modifier = Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    StatusBadge(
                        text = photo.landmarkType.displayName,
                        color = MaterialTheme.colorScheme.primary
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    if (photo.isExifSanitized) {
                        StatusBadge(text = "EXIF Stripped", color = SignalGreen)
                    }
                }

                Spacer(modifier = Modifier.height(4.dp))

                Text(
                    text = photo.displayLocation,
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold
                )

                if (photo.description.isNotBlank()) {
                    Text(
                        text = photo.description,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                Spacer(modifier = Modifier.height(2.dp))

                Text(
                    text = "${photo.filename} • ${photo.fileSizeBytes / 1024} KB • ${photo.imageWidth}x${photo.imageHeight}",
                    style = MaterialTheme.typography.labelSmall,
                    color = Color.Gray
                )
            }

            // Delete action
            IconButton(onClick = onDeleteClick) {
                Icon(
                    imageVector = Icons.Default.Delete,
                    contentDescription = "Delete Photo",
                    tint = SignalRed.copy(alpha = 0.8f)
                )
            }
        }
    }
}
