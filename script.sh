#!/bin/bash
IFS=$' \t\r\n'

curl -X GET \
"https://graph.instagram.com/me?fields=id,username&access_token=IGQWRQRnprb2ExNGVvdFBFRTJRaUFJNF9vdWQwNnkyOFdYU0FYWGxQZAE1KLVpDM09qa2tBT2Y3bk5QVzhnTDlxUnRiZAi1pTTRldmlPVXVtTE9QRGdZAX1FrRS1QbURXcUctQkl6YXY5cHhjZAwZDZD"

curl -X GET \
"https://graph.instagram.com/me/media?fields=id,caption&access_token=IGQWRQRnprb2ExNGVvdFBFRTJRaUFJNF9vdWQwNnkyOFdYU0FYWGxQZAE1KLVpDM09qa2tBT2Y3bk5QVzhnTDlxUnRiZAi1pTTRldmlPVXVtTE9QRGdZAX1FrRS1QbURXcUctQkl6YXY5cHhjZAwZDZD"

curl -X GET \
"https://graph.instagram.com/18001967042105760?fields=id,media_type,media_url,username,timestamp&access_token=IGQWRQRnprb2ExNGVvdFBFRTJRaUFJNF9vdWQwNnkyOFdYU0FYWGxQZAE1KLVpDM09qa2tBT2Y3bk5QVzhnTDlxUnRiZAi1pTTRldmlPVXVtTE9QRGdZAX1FrRS1QbURXcUctQkl6YXY5cHhjZAwZDZD"
