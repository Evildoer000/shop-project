package com.bytedance.shopguide.ui.chat

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.LocalMall
import androidx.compose.material.icons.filled.ShoppingCart
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import coil.compose.SubcomposeAsyncImage
import com.bytedance.shopguide.data.resolveProductImageUrl
import com.bytedance.shopguide.model.ProductCardDto
import com.bytedance.shopguide.ui.theme.MutedInk
import com.bytedance.shopguide.ui.theme.PriceRed
import com.bytedance.shopguide.ui.theme.RatingGold

@Composable
fun ProductCardItem(
    product: ProductCardDto,
    onClick: () -> Unit,
    onAddToCart: (() -> Unit)? = null,
    modifier: Modifier = Modifier,
) {
    Card(
        modifier = modifier
            .fillMaxWidth()
            .wrapContentHeight()
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface,
        ),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.62f)),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.Top,
        ) {
            ProductImage(
                imageUrl = product.imageUrl,
                modifier = Modifier
                    .size(104.dp)
                    .clip(RoundedCornerShape(8.dp)),
            )
            Spacer(Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = product.name,
                    style = MaterialTheme.typography.titleMedium,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                )
                Spacer(Modifier.height(3.dp))
                Text(
                    text = listOfNotNull(product.brand.takeIf { it.isNotBlank() }, product.category)
                        .joinToString(" · "),
                    style = MaterialTheme.typography.bodySmall,
                    color = MutedInk,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                if (product.tags.isNotEmpty()) {
                    Spacer(Modifier.height(8.dp))
                    TagRow(product.tags.take(4))
                }
                if (product.reason.isNotBlank()) {
                    Spacer(Modifier.height(8.dp))
                    Text(
                        text = product.reason.cleanReason(),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.72f),
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                Spacer(Modifier.height(10.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            text = "¥${formatPrice(product.price)}",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = PriceRed,
                        )
                        if (product.rating > 0) {
                            Spacer(Modifier.width(10.dp))
                            Icon(
                                Icons.Filled.Star,
                                contentDescription = null,
                                tint = RatingGold,
                                modifier = Modifier.size(15.dp),
                            )
                            Spacer(Modifier.width(3.dp))
                            Text(
                                text = String.format("%.1f", product.rating),
                                style = MaterialTheme.typography.bodySmall,
                                color = MutedInk,
                            )
                        }
                    }
                    if (onAddToCart != null) {
                        Button(
                            onClick = onAddToCart,
                            shape = RoundedCornerShape(8.dp),
                            contentPadding = androidx.compose.foundation.layout.PaddingValues(
                                horizontal = 10.dp,
                                vertical = 0.dp,
                            ),
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                            modifier = Modifier.height(34.dp),
                        ) {
                            Icon(
                                Icons.Filled.ShoppingCart,
                                contentDescription = null,
                                modifier = Modifier.size(15.dp),
                            )
                            Spacer(Modifier.width(5.dp))
                            Text("加购", style = MaterialTheme.typography.labelMedium)
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun TagRow(tags: List<String>) {
    FlowRow(
        horizontalArrangement = Arrangement.spacedBy(5.dp),
        verticalArrangement = Arrangement.spacedBy(5.dp),
    ) {
        tags.forEach { tag ->
            Surface(
                color = MaterialTheme.colorScheme.surfaceVariant,
                shape = RoundedCornerShape(8.dp),
            ) {
                Text(
                    text = tag,
                    modifier = Modifier.padding(horizontal = 7.dp, vertical = 3.dp),
                    style = MaterialTheme.typography.labelSmall,
                    color = MutedInk,
                )
            }
        }
    }
}

@Composable
fun ProductImage(imageUrl: String, modifier: Modifier = Modifier) {
    val resolvedUrl = resolveProductImageUrl(imageUrl)
    if (resolvedUrl.isBlank()) {
        Placeholder(modifier)
        return
    }
    SubcomposeAsyncImage(
        model = resolvedUrl,
        contentDescription = "商品图片",
        contentScale = ContentScale.Crop,
        modifier = modifier,
        loading = { Placeholder(modifier) },
        error = { Placeholder(modifier) },
    )
}

@Composable
private fun Placeholder(modifier: Modifier = Modifier) {
    Box(
        modifier = modifier.background(MaterialTheme.colorScheme.surfaceVariant),
        contentAlignment = Alignment.Center,
    ) {
        Icon(
            Icons.Filled.LocalMall,
            contentDescription = null,
            tint = MutedInk,
            modifier = Modifier.size(28.dp),
        )
    }
}

private fun String.cleanReason(): String =
    replace(Regex("\\*\\*(.*?)\\*\\*"), "$1")
        .replace(Regex("\\s*\\(\\s*product_id\\s*:\\s*[^)]*\\)", RegexOption.IGNORE_CASE), "")
        .trim()

internal fun formatPrice(price: Double): String =
    if (price % 1.0 == 0.0) price.toLong().toString() else String.format("%.2f", price)
