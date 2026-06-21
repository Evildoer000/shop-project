package com.bytedance.shopguide.data

import android.content.ContentResolver
import android.net.Uri
import android.webkit.MimeTypeMap
import com.bytedance.shopguide.model.ImageUploadResponse
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException

class ImageApiClient(
    baseUrl: String,
    private val json: Json = ApiConfig.json,
    okHttp: OkHttpClient? = null,
) {
    private val uploadUrl = baseUrl.trimEnd('/') + "/api/images"
    private val client: OkHttpClient = okHttp ?: ApiConfig.sharedOkHttp

    suspend fun upload(contentResolver: ContentResolver, uri: Uri): ImageUploadResponse = withContext(Dispatchers.IO) {
        val bytes = contentResolver.openInputStream(uri)?.use { it.readBytes() }
            ?: throw IOException("无法读取图片")
        if (bytes.isEmpty()) throw IOException("图片内容为空")

        val mime = contentResolver.getType(uri) ?: "image/jpeg"
        val ext = MimeTypeMap.getSingleton().getExtensionFromMimeType(mime) ?: "jpg"
        val filename = "upload_${System.currentTimeMillis()}.$ext"
        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                name = "file",
                filename = filename,
                body = bytes.toRequestBody(mime.toMediaTypeOrNull()),
            )
            .build()
        val request = Request.Builder().url(uploadUrl).post(body).build()

        client.newCall(request).execute().use { response ->
            val text = response.body?.string().orEmpty()
            if (!response.isSuccessful) {
                throw IOException("上传失败 ${response.code}: $text")
            }
            json.decodeFromString(ImageUploadResponse.serializer(), text)
        }
    }
}
