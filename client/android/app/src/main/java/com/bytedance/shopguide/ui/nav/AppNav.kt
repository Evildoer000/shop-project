package com.bytedance.shopguide.ui.nav

import androidx.compose.runtime.Composable
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.bytedance.shopguide.ui.cart.CartScreen
import com.bytedance.shopguide.ui.chat.ChatScreen
import com.bytedance.shopguide.ui.mall.MallScreen
import com.bytedance.shopguide.ui.product.ProductDetailScreen

object Routes {
    const val CHAT = "chat"
    const val MALL = "mall"
    const val CART = "cart"
    const val PRODUCT_DETAIL = "product/{productId}"
    fun productDetail(productId: String) = "product/$productId"
}

@Composable
fun AppNav() {
    val navController = rememberNavController()
    NavHost(navController = navController, startDestination = Routes.CHAT) {
        composable(Routes.CHAT) {
            ChatScreen(
                onProductClick = { productId ->
                    navController.navigate(Routes.productDetail(productId))
                },
                onMallClick = {
                    navController.navigate(Routes.MALL)
                },
                onCartClick = {
                    navController.navigate(Routes.CART)
                },
            )
        }
        composable(Routes.MALL) {
            MallScreen(
                onBack = { navController.popBackStack() },
                onProductClick = { productId ->
                    navController.navigate(Routes.productDetail(productId))
                },
            )
        }
        composable(Routes.CART) {
            CartScreen(
                onBack = { navController.popBackStack() },
                onProductClick = { productId ->
                    navController.navigate(Routes.productDetail(productId))
                },
            )
        }
        composable(
            route = Routes.PRODUCT_DETAIL,
            arguments = listOf(navArgument("productId") { type = NavType.StringType }),
        ) { backStackEntry ->
            val productId = backStackEntry.arguments?.getString("productId").orEmpty()
            ProductDetailScreen(
                productId = productId,
                onBack = { navController.popBackStack() },
            )
        }
    }
}
