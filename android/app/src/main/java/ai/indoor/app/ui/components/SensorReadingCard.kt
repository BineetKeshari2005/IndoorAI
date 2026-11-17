package ai.indoor.app.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.indoor.app.domain.models.SensorAccuracy

@Composable
fun SensorReadingCard(
    sensorName: String,
    unit: String,
    x: Float,
    y: Float,
    z: Float,
    magnitude: Float,
    isAvailable: Boolean,
    accuracy: SensorAccuracy = SensorAccuracy.HIGH,
    timestampNs: Long = 0L,
    extraMetrics: @Composable (() -> Unit)? = null,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = sensorName,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.weight(1f)
                )

                if (isAvailable) {
                    val accColor = when (accuracy) {
                        SensorAccuracy.HIGH -> Color(0xFF15803D)
                        SensorAccuracy.MEDIUM -> Color(0xFFB45309)
                        SensorAccuracy.LOW -> Color(0xFFEA580C)
                        SensorAccuracy.UNRELIABLE -> Color(0xFFDC2626)
                    }
                    Text(
                        text = accuracy.label,
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        color = accColor,
                        modifier = Modifier.padding(end = 6.dp)
                    )
                }

                StatusBadge(
                    text = if (isAvailable) "PRESENT" else "NOT DETECTED",
                    backgroundColor = if (isAvailable) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.errorContainer,
                    contentColor = if (isAvailable) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.onErrorContainer
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            Row(modifier = Modifier.fillMaxWidth()) {
                AxisColumn(label = "X", value = x, unit = unit, modifier = Modifier.weight(1f))
                AxisColumn(label = "Y", value = y, unit = unit, modifier = Modifier.weight(1f))
                AxisColumn(label = "Z", value = z, unit = unit, modifier = Modifier.weight(1f))
            }

            Spacer(modifier = Modifier.height(8.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "Magnitude: %+.3f %s".format(magnitude, unit),
                    style = MaterialTheme.typography.bodySmall,
                    fontWeight = FontWeight.Medium,
                    color = MaterialTheme.colorScheme.primary
                )

                if (timestampNs > 0L) {
                    Text(
                        text = "ts: %d ns".format(timestampNs),
                        style = MaterialTheme.typography.labelSmall,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 9.sp,
                        color = Color.Gray
                    )
                }
            }

            if (extraMetrics != null) {
                Spacer(modifier = Modifier.height(8.dp))
                extraMetrics()
            }
        }
    }
}

@Composable
fun OrientationReadingCard(
    azimuthDeg: Float,
    pitchDeg: Float,
    rollDeg: Float,
    hasOrientation: Boolean,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Orientation Estimation (Triad Fusion)",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.weight(1f)
                )

                StatusBadge(
                    text = if (hasOrientation) "CALCULATED" else "WAITING SENSORS",
                    backgroundColor = if (hasOrientation) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant,
                    contentColor = if (hasOrientation) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            Row(modifier = Modifier.fillMaxWidth()) {
                AxisColumn(label = "Yaw (Azimuth)", value = azimuthDeg, unit = "°", modifier = Modifier.weight(1f))
                AxisColumn(label = "Pitch", value = pitchDeg, unit = "°", modifier = Modifier.weight(1f))
                AxisColumn(label = "Roll", value = rollDeg, unit = "°", modifier = Modifier.weight(1f))
            }

            Spacer(modifier = Modifier.height(8.dp))

            val compassHeading = ((azimuthDeg % 360f) + 360f) % 360f
            val cardinalDirection = when {
                compassHeading in 337.5..360.0 || compassHeading in 0.0..22.5 -> "North"
                compassHeading in 22.5..67.5 -> "North-East"
                compassHeading in 67.5..112.5 -> "East"
                compassHeading in 112.5..157.5 -> "South-East"
                compassHeading in 157.5..202.5 -> "South"
                compassHeading in 202.5..247.5 -> "South-West"
                compassHeading in 247.5..292.5 -> "West"
                else -> "North-West"
            }

            Text(
                text = "Compass Heading: %.1f° (%s)".format(compassHeading, cardinalDirection),
                style = MaterialTheme.typography.bodySmall,
                fontWeight = FontWeight.Medium,
                color = MaterialTheme.colorScheme.primary
            )
        }
    }
}

@Composable
private fun AxisColumn(
    label: String,
    value: Float,
    unit: String,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = "%+.2f %s".format(value, unit),
            style = MaterialTheme.typography.bodyMedium,
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.SemiBold
        )
    }
}
