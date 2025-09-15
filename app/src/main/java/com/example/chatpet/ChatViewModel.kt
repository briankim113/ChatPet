// ChatViewModel.kt
package com.example.chatpet

import android.content.Context
import android.util.Log
import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class ChatViewModel : ViewModel() {
    companion object {
        private const val TAG = "ChatViewModel"
    }

    private val _uiState = mutableStateOf<LlmUiState>(LlmUiState.Idle)
    val uiState: State<LlmUiState> = _uiState

    // Keep LlmInference instance if you plan to make multiple calls
    // Ensure it's closed properly in onCleared()
    private var llmInference: LlmInference? = null

    fun generateResponse(context: Context, modelPath: String, prompt: String) {
        if (_uiState.value == LlmUiState.Loading) {
            Log.d(TAG, "Already loading, request ignored.")
            return
        }
        _uiState.value = LlmUiState.Loading
        Log.i(TAG, "Starting LLM response generation for prompt: $prompt")

        viewModelScope.launch {
            try {
                // Initialize LLM Inference if not already done or if it needs to be fresh
                // For simplicity, creating it each time. For performance, you might cache it.
                // Ensure this part is also efficient or handled if llmInference can be reused.
                val taskOptions = LlmInference.LlmInferenceOptions.builder()
                    .setModelPath(modelPath)
                    .setMaxTopK(64) // Add other options as needed (maxTokens, temperature, etc.)
                    .build()

                // createFromOptions can also be blocking
                llmInference = withContext(Dispatchers.IO) {
                    LlmInference.createFromOptions(context, taskOptions)
                }
                Log.d(TAG, "LlmInference instance created/reused.")

                // Actual blocking call on IO dispatcher
                val result = withContext(Dispatchers.IO) {
                    llmInference?.generateResponse(prompt)
                }

                if (result != null) {
                    Log.i(TAG, "LLM Result: $result")
                    _uiState.value = LlmUiState.Success(result)
                } else {
                    Log.e(TAG, "LLM result was null")
                    _uiState.value = LlmUiState.Error("LLM returned no result.")
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error generating LLM response: ${e.message}", e)
                _uiState.value = LlmUiState.Error(e.message ?: "An unknown error occurred")
            } finally {
                // Close the instance if you create it fresh each time,
                // or manage its lifecycle if you reuse it.
                withContext(Dispatchers.IO) {
                    llmInference?.close()
                    llmInference = null // Nullify if creating fresh next time
                }
                Log.d(TAG, "LlmInference instance closed.")
                // Optionally, you might want to revert to Idle state after an error
                // or if the user should be able to make another request.
                // if (_uiState.value is LlmUiState.Error) {
                // _uiState.value = LlmUiState.Idle
                // }
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        viewModelScope.launch(Dispatchers.IO) {
            llmInference?.close() // Ensure it's closed when ViewModel is destroyed
            llmInference = null
            Log.d(TAG, "LlmInference instance closed in onCleared.")
        }
    }
}
