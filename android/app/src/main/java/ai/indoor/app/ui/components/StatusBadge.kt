package ai.indoor.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.indoor.app.ui.theme.MockBadgeColor
import ai.indoor.app.ui.theme.SignalGreen
import ai.indoor.app.ui.theme.SignalRed

@Composable
fun StatusBadge(
    text: String,
    modifier: Modifier = Modifier,
    backgroundColor: Color = MaterialTheme.colorScheme.primaryContainer,
    contentColor: Color = MaterialTheme.colorScheme.onPrimaryContainer
) {
    Box(
        modifier = modifier
            .background(backgroundColor, RoundedCornerShape(12.dp))
            .padding(horizontal = 8.dp, vertical = 4.dp)
    ) {
        Text(
            text = text,
            color = contentColor,
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
fun MockDataBadge(
    modifier: Modifier = Modifier
) {
    StatusBadge(
        text = "DEMONSTRATION DATA",
        modifier = modifier,
        backgroundColor = MockBadgeColor.copy(alpha = 0.15f),
        contentColor = Color(0xFFB45309)
    )
}

@Composable
fun OnlineStatusBadge(
    isOnline: Boolean?,
    modifier: Modifier = Modifier
) {
    when (isOnline) {
        true -> StatusBadge(
            text = "BACKEND ONLINE",
            modifier = modifier,
            backgroundColor = SignalGreen.copy(alpha = 0.15f),
            contentColor = SignalGreen
        )
        false -> StatusBadge(
            text = "BACKEND OFFLINE",
            modifier = modifier,
            backgroundColor = SignalRed.copy(alpha = 0.15f),
            contentColor = SignalRed
        )
        null -> StatusBadge(
            text = "CHECKING...",
            modifier = modifier,
            backgroundColor = Color.LightGray.copy(alpha = 0.3f),
            contentColor = Color.DarkGray
        )
    }
}

