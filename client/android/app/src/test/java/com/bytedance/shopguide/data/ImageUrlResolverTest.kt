package com.bytedance.shopguide.data

import org.junit.Assert.assertEquals
import org.junit.Test

class ImageUrlResolverTest {
    @Test
    fun absoluteUrlReturnsUnchanged() {
        assertEquals(
            "https://example.com/p.png",
            resolveProductImageUrl("https://example.com/p.png", "http://10.0.2.2:8000/"),
        )
    }

    @Test
    fun relativeDatasetPathUsesBaseUrl() {
        assertEquals(
            "http://10.0.2.2:8000/dataset/products/p.png",
            resolveProductImageUrl("/dataset/products/p.png", "http://10.0.2.2:8000/"),
        )
    }

    @Test
    fun relativeUploadsPathUsesBaseUrl() {
        assertEquals(
            "http://10.0.2.2:8000/uploads/p.png",
            resolveProductImageUrl("uploads/p.png", "http://10.0.2.2:8000/"),
        )
    }

    @Test
    fun blankUrlStaysBlank() {
        assertEquals("", resolveProductImageUrl("   ", "http://10.0.2.2:8000/"))
    }
}
