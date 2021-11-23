(function( exports ) {
    "use strict";

    var version = 0.1,
    params = {
        COOKIE_NAME: "litw_locale",
        COOKIE_EXPIRATION: function() {
            var date = new Date();
            return date.setTime(date.getTime() + (365*24*60*60*1000)); // one year
        }(),
        set: false,
        defaultLocale: "en",
        locale: null
    },

    _checkQueryString = function() {

        // check if a locale is set in the query string
        if (!params.set) {

            var url = new URL(window.location.href);
            var url_locale = url.searchParams.get("locale");
            if (url_locale) {
                params.locale = url_locale;
                params.set = true;
            }
        }
    },

    _checkCookies = function() {

        // check if a locale cookie exists
        if (!params.set) {
            $.each(document.cookie.split(";"), function() {
                var cookie = this.split("=");
                if (cookie[0] === params.COOKIE_NAME) {
                    params.locale = cookie[1];
                    params.set = true;
                }
            });
        }
    },

    _checkBrowserLang = function() {

        // attempt to get the browser's default language
        if (!params.set) {

            // NOTE: use navigator.languages for Chrome, which will report the
            // installation language (not the browser language preference) under
            // navigator.language
            var browserLocale = (navigator.languages) ?
                navigator.languages[0] :
                (navigator.language || navigator.userLanguage);

            if (browserLocale) {

                // simplify a locale of the form "en-US" to "en"
                browserLocale = browserLocale.split("-")[0];

                // set cookie using the browser locale value
                document.cookie = params.COOKIE_NAME + "=" + browserLocale +
                    "; expires=" + params.COOKIE_EXPIRATION + "; path=/";

                params.locale = browserLocale;
                params.set = true;
            } else {

                // use the default locale value
                params.locale = params.defaultLocale;
                params.set = true;
            }
        }
    },

    getLocale = function() {

        // Determine the locale to used, as follows:
        // 1. if a locale query parameter is passed, use this and ignore other settings
        // 2. otherwise, if a cookie is found, use this value
        // 3. otherwise, try to get the browser language and set the cookie with this value
        // 4, otherwise, use the default locale
        _checkQueryString();
        _checkCookies();
        _checkBrowserLang();

        return params.locale;
    }

    exports.locale = {};
    exports.locale.getLocale = getLocale;

})( window.LITW = window.LITW || {} );