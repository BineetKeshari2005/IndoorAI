package ai.indoor.app.viewmodel

import ai.indoor.app.camera.CameraManager
import ai.indoor.app.camera.ImageCaptureCallback
import ai.indoor.app.camera.privacy.FaceBounds
import ai.indoor.app.domain.models.FaceDetectionStatus
import ai.indoor.app.domain.models.LandmarkType
import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.mockito.Mockito.mock
import java.io.File

@OptIn(ExperimentalCoroutinesApi::class)
class ImageCaptureViewModelTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    private val testDispatcher = StandardTestDispatcher()

    @Before
    fun setUp() {
        Dispatchers.setMain(testDispatcher)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    private class FakeCameraManager(
        var available: Boolean = true
    ) : CameraManager {
        var isPreviewRunning = false
        var captureInvoked = false

        override fun isCameraAvailable(): Boolean = available

        override fun startPreview(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
            isPreviewRunning = true
        }

        override fun stopPreview() {
            isPreviewRunning = false
        }

        override fun capturePhoto(callback: ImageCaptureCallback) {
            captureInvoked = true
        }
    }

    @Test
    fun testInitialStateCameraIsStrictlyInactive() {
        val fakeCamera = FakeCameraManager()
        val viewModel = ImageCaptureViewModel(cameraManager = fakeCamera)

        // Strict requirement: Preview must NEVER start automatically
        assertEquals(PreviewState.INACTIVE, viewModel.previewState.value)
        assertEquals(CaptureFlowState.Idle, viewModel.captureFlowState.value)
        assertTrue("Face blur should be enabled by default", viewModel.faceBlurEnabled.value)
        assertFalse(fakeCamera.isPreviewRunning)
    }

    @Test
    fun testStartPreviewRequiresPermission() {
        val fakeCamera = FakeCameraManager()
        val viewModel = ImageCaptureViewModel(cameraManager = fakeCamera)
        val mockLifecycleOwner = mock(LifecycleOwner::class.java)
        val mockPreviewView = mock(PreviewView::class.java)

        viewModel.startCameraPreview(mockLifecycleOwner, mockPreviewView, hasPermission = false)

        assertEquals(PreviewState.PERMISSION_REQUIRED, viewModel.previewState.value)
        assertFalse("Camera preview should not be active without permission", fakeCamera.isPreviewRunning)
    }

    @Test
    fun testCameraUnavailableTransitionsCorrectly() {
        val unavailableCamera = FakeCameraManager(available = false)
        val viewModel = ImageCaptureViewModel(cameraManager = unavailableCamera)
        val mockLifecycleOwner = mock(LifecycleOwner::class.java)
        val mockPreviewView = mock(PreviewView::class.java)

        viewModel.startCameraPreview(mockLifecycleOwner, mockPreviewView, hasPermission = true)

        assertEquals(PreviewState.UNAVAILABLE, viewModel.previewState.value)
        assertFalse(unavailableCamera.isPreviewRunning)
    }

    @Test
    fun testStopPreviewReleasesCamera() {
        val fakeCamera = FakeCameraManager(available = true)
        val viewModel = ImageCaptureViewModel(cameraManager = fakeCamera)

        // Simulate stopping
        viewModel.stopCameraPreview()
        assertEquals(PreviewState.INACTIVE, viewModel.previewState.value)
        assertFalse(fakeCamera.isPreviewRunning)
    }

    @Test
    fun testUpdateDraftAnnotations() {
        val viewModel = ImageCaptureViewModel()

        viewModel.updateDraft(
            building = "TI-Engineering",
            floor = "3",
            room = "Room 304",
            landmarkType = LandmarkType.ROOM_SIGN,
            description = "Office Door Plaque",
            notes = "Direct sunlight from left"
        )

        val draft = viewModel.currentDraft.value
        assertEquals("TI-Engineering", draft.building)
        assertEquals("3", draft.floor)
        assertEquals("Room 304", draft.room)
        assertEquals(LandmarkType.ROOM_SIGN, draft.landmarkType)
        assertEquals("Office Door Plaque", draft.description)
        assertEquals("Direct sunlight from left", draft.notes)
    }

    @Test
    fun testToggleFaceBlur() {
        val viewModel = ImageCaptureViewModel()

        assertTrue(viewModel.faceBlurEnabled.value)

        viewModel.toggleFaceBlur(false)
        assertFalse(viewModel.faceBlurEnabled.value)

        viewModel.toggleFaceBlur(true)
        assertTrue(viewModel.faceBlurEnabled.value)
    }

    @Test
    fun testRetakePhotoDeletesTempFileAndResetsToIdle() {
        val viewModel = ImageCaptureViewModel()
        val tempRawFile = tempFolder.newFile("test_raw.jpg")
        assertTrue(tempRawFile.exists())

        // Set state to reviewing
        val reviewState = CaptureFlowState.Reviewing(
            rawTempFile = tempRawFile,
            previewBitmap = null,
            detectedFacesCount = 0,
            detectedFaces = emptyList(),
            faceStatus = FaceDetectionStatus.NO_FACES_DETECTED,
            isBlurEnabled = true,
            draft = LandmarkMetadataDraft()
        )

        // Inject reviewing state reflection or direct flow transition simulation
        val field = ImageCaptureViewModel::class.java.getDeclaredField("_captureFlowState")
        field.isAccessible = true
        @Suppress("UNCHECKED_CAST")
        val stateFlow = field.get(viewModel) as kotlinx.coroutines.flow.MutableStateFlow<CaptureFlowState>
        stateFlow.value = reviewState

        assertEquals(reviewState, viewModel.captureFlowState.value)

        // Call retake
        viewModel.retakePhoto()

        assertEquals(CaptureFlowState.Idle, viewModel.captureFlowState.value)
        assertFalse("Temporary raw file must be deleted upon retake", tempRawFile.exists())
    }

    @Test
    fun testResetCaptureState() {
        val viewModel = ImageCaptureViewModel()
        viewModel.resetCaptureState()
        assertEquals(CaptureFlowState.Idle, viewModel.captureFlowState.value)
    }
}
