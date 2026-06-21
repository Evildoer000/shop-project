package com.bytedance.shopguide

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.bytedance.shopguide.ui.nav.AppNav
import com.bytedance.shopguide.ui.theme.ShopGuideTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            ShopGuideTheme {
                AppNav()
            }
        }
    }
}
