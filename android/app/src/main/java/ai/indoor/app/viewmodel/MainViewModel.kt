package ai.indoor.app.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import ai.indoor.app.data.repository.LocalizationRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * Global application ViewModel managing backend connection settings and health status.
 */
class MainViewModel(
    private val repository: LocalizationRepository
) : ViewModel() {

    private val _backendUrl = MutableStateFlow(repository.getBaseUrl())
    val backendUrl: StateFlow<String> = _backendUrl.asStateFlow()

    private val _isServerOnline = MutableStateFlow<Boolean?>(null)
    val isServerOnline: StateFlow<Boolean?> = _isServerOnline.asStateFlow()

    private val _isCheckingHealth = MutableStateFlow(false)
    val isCheckingHealth: StateFlow<Boolean> = _isCheckingHealth.asStateFlow()

    init {
        checkServerHealth()
    }

    fun updateBackendUrl(newUrl: String) {
        _backendUrl.value = newUrl
        repository.updateBaseUrl(newUrl)
        checkServerHealth()
    }

    fun checkServerHealth() {
        viewModelScope.launch {
            _isCheckingHealth.value = true
            val result = repository.checkServerHealth()
            _isServerOnline.value = result.getOrDefault(false)
            _isCheckingHealth.value = false
        }
    }
}

