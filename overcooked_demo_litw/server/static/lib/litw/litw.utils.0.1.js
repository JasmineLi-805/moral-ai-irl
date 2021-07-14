/*************************************************************
 * litw.utils.0.1.js.php
 * 
 * Contains utility methods for LITW experiments. 
 *
 * Dependencies: jQuery
 * 
 * Author: Trevor Croxson
 *
 * Last Modified: January 28, 2017
 * 
 * © Copyright 2017 LabintheWild
 * For questions about this file and permission to use
 * the code, contact us at info@labinthewild.org
 *************************************************************/

//module.exports = exports = (function( exports ) {
(function( exports ) {
	"use strict";

	var version = "0.1",
	defaultProperties = {
		// the next button goes by a few different names
		nextBtnSelectors: "#next_button, #next-button, #btn-next-page, .next-button, .btn-next-page",
		// keys which, when pressed, will trigger the next button.
		// such keys will not trigger the next button when the page
		// focus is on any text entry element, as defined below
		submitKeys: [
			32 // spacebar
		],
		textEntryElts: "textarea, input[type='text'], input.custom-combobox-input",
		ajaxWorkingSelectors: "#ajax-working, #ajaxWorking",
		fadeElt: "#content",
		loadingTime: 1000,
		useLoadingEffect: true,
		hideOnComplete: true,
		validator: null
	},
	properties = {},

	/**
	 * Display the next button. When clicked, fades the fadeElt and
	 * displays the spinner gif.
	 * 
	 * @function showNextButton
	 * @param {Function} fn 															The function to call when the next button is pressed
	 * @param {Object} [props] 														Optional properties object
	 * @param {string} [props.fadeElt="#content"] 				CSS selector of element to fade out when the next button
	 *                                                		is clicked
	 * @param {boolean} [props.useLoadingEffect=true] 		Whether or not to fade out properties.fadeElet, show
	 *                                                  	the ajax spinner, and wait properties.loadingTime milliseconds
	 *                                                  	before advancing
	 * @param {array} [props.submitKeys] 									Keycodes for keys that, when pressed, will trigger the next button
	 * @param {string} [props.textEntryElts] 							String of selectors of elements that accept text entry; when the
	 *                                           					focus is on these elements, pressing any one of props.submitKeys will
	 *                                           					not trigger the next button
	 * @param {number} [props.loadingTime=1000] 					The amount of time, in milliseconds, to delay before
	 *                                               			executing fn()
	 * @param {boolean} [props.hideOnComplete=true] 			Whether or not to hide the next button after fn() is executed
	 * @param {function} [props.validator] 								A reference to an optional function to run when the next button is
	 *                                             				clicked. If this function returns a falsy value, the next button's
	 *                                             				callback will not be executed. Useful for performing form validation, etc.
	 */
	showNextButton = function(fn, props) {
		// listen for events triggered by the combobox plugin, since it
		// is likely that the plugin (if in use) will render fresh text 
		// input elements after the util module's initial focus/focusout 
		// listeners have been bound
		$(window).on("combobox:render", function() {
			_bindTextEltFocusListeners();
		});

		properties.fn = fn;
		_updateProperties(props);
		_bindKeyupListeners();
		_bindTextEltFocusListeners();

		if (!$(properties.ajaxWorkingSelectors).children("img").length) {
			$(properties.ajaxWorkingSelectors)
				.html("<img src='http://static.labinthewild.org/images/ajax-loader.gif' />")
				.hide();
		}

		$(properties.nextBtnSelectors)
			.show()
			.off()
			.on("click", _onClickNextBtn);
	},

	_updateProperties = function(propsToUpdate) {
		var propsToUpdate = propsToUpdate || {};
		$.extend(true, properties, defaultProperties);
		for (var prop in propsToUpdate) {
			properties[prop] = propsToUpdate[prop];
		}
	},

	_onClickNextBtn = function() {
		if (properties.validator &&
			typeof properties.validator === "function") {
			var validatedResponse = properties.validator();

			if (validatedResponse) {
				_proceed(validatedResponse);
			}
		} else {
			_proceed();
		}
	},

	_proceed = function(validatedResponse) {
		_unbindAllListeners();

		if (properties.useLoadingEffect) {
			showLoadingIcon();
		}
		window.setTimeout(function() {
			if (properties.hideOnComplete) {
				hideNextButton();						
			}
			hideLoadingIcon();
			properties.fn(validatedResponse);
		}, ((properties.useLoadingEffect) ? properties.loadingTime : 0));
	},

	/**
	 * Hide the next button.
	 * 
	 * @function hideNextButton
	 */
	hideNextButton = function() {
		_unbindAllListeners();
		$(properties.nextBtnSelectors)
			.hide();
	},

	_unbindAllListeners = function() {
		$(window)
			.off("keyup");
		$(properties.nextBtnSelectors)
			.off("click");
		$(properties.textEntryElts)
			.off("focus focusout");
	},

	_bindTextEltFocusListeners = function() {
		$(properties.textEntryElts)
			.off("focus focusout")
			.focus(function() {
				// remove keyup listeners so keypresses in text
				// entry fields don't trigger a next button click
				$(window).off("keyup");
			})
			.focusout(function() {
				_bindKeyupListeners();
			});
	},

	_bindKeyupListeners = function() {
		$(window).keyup(function(e) {
      if ($.inArray(e.which, properties.submitKeys) > -1) {
        _onClickNextBtn();
      }
    });
	},

	/**
	 * Fades out properties.fadeElt and displays the ajax spinner
	 * 
	 * @function showLoadingIcon
	 * @param {Object} [props]  													Optional properties object
	 * @param {string} [props.fadeElt="#content"] 				CSS selector of element to fade out
	 */
	showLoadingIcon = function(props) {
		var props = props || {};
		properties.fadeElt = props.fadeElt || properties.fadeElt;
		$(properties.fadeElt).css("opacity", "0.5");
		$(properties.ajaxWorkingSelectors).show();
	},

	/**
	 * Hides the ajax spinner and restores properties.fadeElt
	 * to full opacity.
	 * 
	 * @function hideLoadingIcon
	 */
	hideLoadingIcon = function() {
		$(properties.fadeElt).css("opacity", "1.0");
		$(properties.ajaxWorkingSelectors).hide();
	},

	/**
	 * Shows an element on the page with class "slide" and id 
	 * passed in. All other elements with class "slide" will be 
	 * hidden. NOTE: code adapted from the zen.js library.
	 * 
	 * @function showSlide
	 * @param {string} id  															The HTML id of the element to show
	 */
	showSlide = function(id) {
		$(properties.ajaxWorkingSelectors).hide();
		properties._currentSlide = id;

		properties._slides = properties._slides || $(".slide");
		var i = properties._slides.length,
		changeTo;

		while (i--) {
			changeTo = ($(properties._slides[i]).attr("id")) == id ? 'block' : 'none';
	 		if ($(properties._slides[i]).css("style", "display") == changeTo) continue;
			$(properties._slides[i]).css("display", changeTo);
		}
	},

	// shuffle any number of equal-length arrays in the same way.
	// returns the shuffled arrays as an array of arrays, unless
	// only a single array is passed, in which case just that array
	// will be returned.
	shuffleArrays = function( /* arrays */ ) {
		var arrays = Array.prototype.slice.call(arguments),
  	currentIndex = arrays[0].length, temporaryValue, randomIndex;
	  while (currentIndex !== 0) {
	    randomIndex = Math.floor(Math.random() * currentIndex);
	    currentIndex--;
	    arrays.forEach(function(array) {
	    	temporaryValue = array[currentIndex];
		    array[currentIndex] = array[randomIndex];
		    array[randomIndex] = temporaryValue;
	    });
	  }

	  if (arrays.length === 1) {
	  	return arrays[0];
	  } else {
	  	return arrays;
	  }
	};

	/**** PUBLIC METHODS ****/
	exports.utils = {};
	exports.utils.showNextButton = showNextButton;
	exports.utils.hideNextButton = hideNextButton;
	exports.utils.showSlide = showSlide;
	exports.utils.showLoadingIcon = showLoadingIcon;
	exports.utils.hideLoadingIcon = hideLoadingIcon;
	exports.utils.shuffleArrays = shuffleArrays;

 })( window.LITW = window.LITW || {} );
