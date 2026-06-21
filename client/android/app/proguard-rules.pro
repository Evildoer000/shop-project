# Default proguard rules.
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.AnnotationsKt

# Keep kotlinx-serialization generated serializers
-keepclassmembers class **$$serializer {
    *** descriptor;
}
-keepclassmembers class * {
    @kotlinx.serialization.Serializable *;
}
